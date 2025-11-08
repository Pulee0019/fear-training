import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import struct
import re
import os
import math
from datetime import datetime
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class FearConditioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Conditioning Analyzer with Fiber Photometry")
        self.root.geometry("1200x800")
        self.root.state('zoomed')  # Start maximized
        
        # Data storage
        self.freezing_data = {}
        self.pupil_data = {}
        self.ast2_data = {}
        self.aligned_data = {}
        self.fiber_data = None
        self.eztrack_results_path = ""
        self.dlc_results_path = ""
        self.timestamp_path = ""
        self.ast2_path = ""
        self.events_path = ""
        self.fiber_data_path = ""
        self.video_start_fiber = None
        self.video_end_fiber = None
        self.cam_id = None
        self.experiment_start = None
        self.event_data = None
        self.window_freeze = None
        self.processed_fiber_data = None
        self.dff_data = None
        self.zscore_data = None
        self.baseline_data = None
        self.motion_corrected = False
        self.baseline_corrected = False
        self.smoothed = False
        self.analysis_mode = None  # 'freezing', 'freezing+fiber', 'pupil', 'pupil+fiber'
        
        # Create main notebook for different analysis modes
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # Create initial selection frame
        self.selection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.selection_frame, text="Select Analysis")
        
        # Create analysis frames
        self.freezing_frame = ttk.Frame(self.notebook)
        self.freezing_fiber_frame = ttk.Frame(self.notebook)
        self.pupil_frame = ttk.Frame(self.notebook)
        self.pupil_fiber_frame = ttk.Frame(self.notebook)
        
        # Add analysis frames to the notebook
        self.notebook.add(self.freezing_frame, text="Freezing Analysis")
        self.notebook.add(self.freezing_fiber_frame, text="Freezing + Fiber Photometry")
        self.notebook.add(self.pupil_frame, text="Pupil Analysis")
        self.notebook.add(self.pupil_fiber_frame, text="Pupil + Fiber Photometry")
        
        # Hide analysis frames initially
        self.notebook.hide(self.freezing_frame)
        self.notebook.hide(self.freezing_fiber_frame)
        self.notebook.hide(self.pupil_frame)
        self.notebook.hide(self.pupil_fiber_frame)
        
        # Initialize frames
        self.create_selection_ui()
        self.create_freezing_ui()
        self.create_freezing_fiber_ui()
        self.create_pupil_ui()
        self.create_pupil_fiber_ui()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
    
    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def create_selection_ui(self):
        """Create initial selection UI with two large buttons"""
        container = ttk.Frame(self.selection_frame)
        container.pack(expand=True, fill="both")
        
        # Add title
        title_label = ttk.Label(container, text="Fear Conditioning Analyzer", font=("Arial", 24, "bold"))
        title_label.pack(pady=40)
        
        # Add subtitle
        subtitle_label = ttk.Label(container, text="Select Analysis Mode", font=("Arial", 16))
        subtitle_label.pack(pady=10)
        
        # Button frame
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=50)
        
        # Freezing analysis button
        freezing_btn = ttk.Button(btn_frame, text="Freezing Analysis", 
                                 command=lambda: self.select_analysis("freezing"),
                                 width=20, style="Large.TButton")
        freezing_btn.grid(row=0, column=0, padx=20, pady=20)
        
        # Freezing with fiber button
        freezing_fiber_btn = ttk.Button(btn_frame, text="Freezing + Fiber Photometry", 
                                       command=lambda: self.select_analysis("freezing+fiber"),
                                       width=20, style="Large.TButton")
        freezing_fiber_btn.grid(row=0, column=1, padx=20, pady=20)
        
        # Pupil analysis button
        pupil_btn = ttk.Button(btn_frame, text="Pupil Analysis", 
                              command=lambda: self.select_analysis("pupil"),
                              width=20, style="Large.TButton")
        pupil_btn.grid(row=1, column=0, padx=20, pady=20)
        
        # Pupil with fiber button
        pupil_fiber_btn = ttk.Button(btn_frame, text="Pupil + Fiber Photometry", 
                                    command=lambda: self.select_analysis("pupil+fiber"),
                                    width=20, style="Large.TButton")
        pupil_fiber_btn.grid(row=1, column=1, padx=20, pady=20)
        
        # Configure style for large buttons
        style = ttk.Style()
        style.configure("Large.TButton", font=("Arial", 14), padding=10)
        
        # Add footer
        footer_label = ttk.Label(container, text="Developed for Neuroscience Research", font=("Arial", 10))
        footer_label.pack(side="bottom", pady=20)
    
    def select_analysis(self, mode):
        """Select analysis mode and switch to appropriate tab"""
        self.analysis_mode = mode
        self.set_status(f"Selected analysis mode: {mode}")
        
        if mode == "freezing":
            self.notebook.select(self.freezing_frame)
        elif mode == "freezing+fiber":
            self.notebook.select(self.freezing_fiber_frame)
        elif mode == "pupil":
            self.notebook.select(self.pupil_frame)
        elif mode == "pupil+fiber":
            self.notebook.select(self.pupil_fiber_frame)
    
    def create_freezing_ui(self):
        """Create UI for freezing analysis without fiber"""
        # Create paned window for left/right split
        paned = ttk.PanedWindow(self.freezing_frame, orient="horizontal")
        paned.pack(fill="both", expand=True)
        
        # Left panel for controls
        control_frame = ttk.Frame(paned, width=300)
        paned.add(control_frame, weight=0)
        
        # Right panel for plots
        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, weight=1)
        
        # Store plot frame for later use
        self.freezing_plot_frame = plot_frame
        
        # Create notebook for steps
        steps_notebook = ttk.Notebook(control_frame)
        steps_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Step 1: Data loading
        step1_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step1_frame, text="1. Data Loading")
        
        # Step 2: Analysis
        step2_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step2_frame, text="2. Analysis")
        
        # Step 3: Visualization
        step3_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step3_frame, text="3. Visualization")
        
        # Step 1 content
        ttk.Label(step1_frame, text="Load ezTrack Results:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_eztrack_results).pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="Load Events CSV:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_events).pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="Load Timestamps:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_timestamps).pack(fill="x", pady=5)
        
        # Step 2 content
        ttk.Label(step2_frame, text="Analysis Parameters").pack(pady=(10, 5), anchor="w")
        
        param_frame = ttk.Frame(step2_frame)
        param_frame.pack(fill="x", pady=5)
        
        ttk.Label(param_frame, text="Window Size (s):").grid(row=0, column=0, sticky="w")
        self.window_size_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.window_size_var, width=10).grid(row=0, column=1, sticky="e")
        
        ttk.Label(param_frame, text="FPS:").grid(row=1, column=0, sticky="w")
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.fps_var, width=10).grid(row=1, column=1, sticky="e")
        
        ttk.Button(step2_frame, text="Run Freezing Analysis", command=self.analyze).pack(pady=10)
        
        # Step 3 content
        ttk.Button(step3_frame, text="Plot Freezing Timeline", command=self.plot_freezing_windows).pack(pady=5)
        ttk.Button(step3_frame, text="Export Results to CSV", command=self.export_freezing_csv).pack(pady=5)
    
    def create_freezing_fiber_ui(self):
        """Create UI for freezing analysis with fiber photometry"""
        # Create paned window for left/right split
        paned = ttk.PanedWindow(self.freezing_fiber_frame, orient="horizontal")
        paned.pack(fill="both", expand=True)
        
        # Left panel for controls
        control_frame = ttk.Frame(paned, width=300)
        paned.add(control_frame, weight=0)
        
        # Right panel for plots
        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, weight=1)
        
        # Store plot frame for later use
        self.freezing_fiber_plot_frame = plot_frame
        
        # Create notebook for steps
        steps_notebook = ttk.Notebook(control_frame)
        steps_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Step 1: Data loading and alignment
        step1_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step1_frame, text="1. Data Loading & Alignment")
        
        # Step 2: Preprocessing
        step2_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step2_frame, text="2. Preprocessing")
        
        # Step 3: ΔF/F & Z-score
        step3_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step3_frame, text="3. ΔF/F & Z-score")
        
        # Step 4: Event Analysis
        step4_frame = ttk.Frame(steps_notebook)
        steps_notebook.add(step4_frame, text="4. Event Analysis")
        
        # Step 1 content
        ttk.Label(step1_frame, text="Load Fiber Data:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_fiber_data).pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="Load ezTrack Results:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_eztrack_results).pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="Load Events CSV:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_events).pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="Load Timestamps:").pack(pady=(10, 5), anchor="w")
        ttk.Button(step1_frame, text="Browse", command=self.load_timestamps).pack(fill="x", pady=5)
        
        ttk.Button(step1_frame, text="Set Video Markers", command=self.set_video_markers).pack(pady=10)
        ttk.Button(step1_frame, text="Align Data", command=self.align_fiber_data).pack(pady=5)
        ttk.Button(step1_frame, text="Plot Raw Fiber Data", command=self.plot_fiber_data).pack(pady=5)
        
        # Step 2 content
        ttk.Label(step2_frame, text="Smoothing").pack(pady=(10, 5), anchor="w")
        
        smooth_frame = ttk.Frame(step2_frame)
        smooth_frame.pack(fill="x", pady=5)
        
        ttk.Label(smooth_frame, text="Window Size (s):").grid(row=0, column=0, sticky="w")
        self.smooth_window_var = tk.StringVar(value="1")
        ttk.Entry(smooth_frame, textvariable=self.smooth_window_var, width=10).grid(row=0, column=1, sticky="e")
        
        ttk.Button(step2_frame, text="Apply Smoothing", command=self.apply_smoothing).pack(pady=5)
        
        ttk.Separator(step2_frame, orient="horizontal").pack(fill="x", pady=10)
        
        ttk.Label(step2_frame, text="Baseline Correction").pack(pady=(10, 5), anchor="w")
        self.baseline_method = tk.StringVar(value="none")
        ttk.Radiobutton(step2_frame, text="None", variable=self.baseline_method, value="none").pack(anchor="w")
        ttk.Radiobutton(step2_frame, text="Exponential Fit", variable=self.baseline_method, value="exp").pack(anchor="w")
        ttk.Radiobutton(step2_frame, text="Linear Fit", variable=self.baseline_method, value="linear").pack(anchor="w")
        ttk.Radiobutton(step2_frame, text="Polynomial Fit", variable=self.baseline_method, value="poly").pack(anchor="w")
        
        ttk.Label(step2_frame, text="Polynomial Degree:").pack(pady=(5, 0), anchor="w")
        self.poly_degree_var = tk.StringVar(value="3")
        ttk.Entry(step2_frame, textvariable=self.poly_degree_var, width=10).pack(anchor="w", padx=20)
        
        ttk.Button(step2_frame, text="Apply Baseline Correction", command=self.apply_baseline_correction).pack(pady=10)
        
        ttk.Separator(step2_frame, orient="horizontal").pack(fill="x", pady=10)
        
        ttk.Label(step2_frame, text="Motion Correction").pack(pady=(10, 5), anchor="w")
        self.motion_method = tk.StringVar(value="none")
        ttk.Radiobutton(step2_frame, text="None", variable=self.motion_method, value="none").pack(anchor="w")
        ttk.Radiobutton(step2_frame, text="Use 410nm Reference", variable=self.motion_method, value="isos").pack(anchor="w")
        ttk.Radiobutton(step2_frame, text="Use Baseline Period", variable=self.motion_method, value="baseline").pack(anchor="w")
        
        ttk.Label(step2_frame, text="Baseline Start (s):").pack(pady=(5, 0), anchor="w")
        self.baseline_start_var = tk.StringVar(value="0")
        ttk.Entry(step2_frame, textvariable=self.baseline_start_var, width=10).pack(anchor="w", padx=20)
        
        ttk.Label(step2_frame, text="Baseline End (s):").pack(pady=(5, 0), anchor="w")
        self.baseline_end_var = tk.StringVar(value="60")
        ttk.Entry(step2_frame, textvariable=self.baseline_end_var, width=10).pack(anchor="w", padx=20)
        
        ttk.Button(step2_frame, text="Apply Motion Correction", command=self.apply_motion_correction).pack(pady=10)
        
        # Step 3 content
        ttk.Label(step3_frame, text="ΔF/F Calculation").pack(pady=(10, 5), anchor="w")
        
        self.dff_method = tk.StringVar(value="standard")
        ttk.Radiobutton(step3_frame, text="Standard ΔF/F", variable=self.dff_method, value="standard").pack(anchor="w")
        ttk.Radiobutton(step3_frame, text="Z-score", variable=self.dff_method, value="zscore").pack(anchor="w")
        
        ttk.Label(step3_frame, text="F0 Calculation Window (s):").pack(pady=(5, 0), anchor="w")
        self.f0_window_var = tk.StringVar(value="60")
        ttk.Entry(step3_frame, textvariable=self.f0_window_var, width=10).pack(anchor="w", padx=20)
        
        ttk.Button(step3_frame, text="Calculate ΔF/F", command=self.calculate_dff).pack(pady=10)
        ttk.Button(step3_frame, text="Plot ΔF/F", command=self.plot_dff).pack(pady=5)
        
        # Step 4 content
        ttk.Label(step4_frame, text="Event Analysis").pack(pady=(10, 5), anchor="w")
        
        ttk.Label(step4_frame, text="Pre-event Time (s):").pack(pady=(5, 0), anchor="w")
        self.pre_event_var = tk.StringVar(value="10")
        ttk.Entry(step4_frame, textvariable=self.pre_event_var, width=10).pack(anchor="w", padx=20)
        
        ttk.Label(step4_frame, text="Post-event Time (s):").pack(pady=(5, 0), anchor="w")
        self.post_event_var = tk.StringVar(value="20")
        ttk.Entry(step4_frame, textvariable=self.post_event_var, width=10).pack(anchor="w", padx=20)
        
        ttk.Button(step4_frame, text="Plot Event Responses", command=self.plot_event_responses).pack(pady=10)
        ttk.Button(step4_frame, text="Plot Event Heatmap", command=self.plot_event_heatmap).pack(pady=5)
        ttk.Button(step4_frame, text="Export Event Data", command=self.export_event_data).pack(pady=5)
    
    def create_pupil_ui(self):
        """Create UI for pupil analysis without fiber"""
        # Similar to freezing_ui but for pupil analysis
        # Implementation would follow similar pattern as freezing_ui
        pass
    
    def create_pupil_fiber_ui(self):
        """Create UI for pupil analysis with fiber photometry"""
        # Similar to freezing_fiber_ui but for pupil analysis
        pass
    
    def clear_plot_frame(self, frame):
        """Clear the plot frame"""
        for widget in frame.winfo_children():
            widget.destroy()
    
    def create_plot_canvas(self, frame, fig=None):
        """Create a matplotlib canvas in the given frame"""
        self.clear_plot_frame(frame)
        
        if fig is None:
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data to display", 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=ax.transAxes,
                    fontsize=14)
            ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        return fig, canvas
    
    # Data loading methods
    def load_fiber_data(self):
        """Load fiber photometry data from CSV file with header handling"""
        self.fiber_data_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.fiber_data_path:
            try:
                self.fiber_data = pd.read_csv(self.fiber_data_path, skiprows=1, delimiter=',')
                
                self.fiber_data.columns = self.fiber_data.columns.str.strip()
                
                time_col = None
                possible_time_columns = ['timestamp', 'timems', 'time', 'time(ms)']
                for col in self.fiber_data.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in possible_time_columns):
                        time_col = col
                        break
                
                isos_col = None
                green_col = None
                red_col = None
                events_col = None
                
                for col in self.fiber_data.columns:
                    col_lower = col.lower()
                    
                    if '410' in col_lower or '415' in col_lower or 'isos' in col_lower:
                        isos_col = col
                    
                    elif '470' in col_lower or 'gcamp' in col_lower or 'green' in col_lower:
                        green_col = col
                    
                    elif '560' in col_lower or 'rcamp' in col_lower or 'red' in col_lower:
                        red_col = col
                    
                    elif 'event' in col_lower:
                        events_col = col
                
                if not time_col:
                    numeric_cols = self.fiber_data.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0 and self.fiber_data[numeric_cols[0]].dtype in [np.int64, np.float64]:
                        time_col = numeric_cols[0]
                        self.set_status(f"Using first numeric column as time: {time_col}")
                    else:
                        messagebox.showerror("Error", "Timestamp column not found in fiber data file")
                        self.fiber_data = None
                        return
                
                if not isos_col:
                    messagebox.showerror("Error", "Isosbestic (410/415nm) column not found")
                    self.fiber_data = None
                    return
                
                if self.fiber_data[time_col].max() > 10000:
                    self.fiber_data[time_col] = self.fiber_data[time_col] / 1000.0
                
                self.fiber_data['normalized_signal'] = np.nan
                
                if green_col:
                    self.fiber_data['normalized_signal_470'] = self.fiber_data[green_col] / self.fiber_data[isos_col]
                    self.fiber_data['normalized_signal'] = self.fiber_data['normalized_signal_470']
                    self.primary_signal = '470nm'
                
                if red_col:
                    self.fiber_data['normalized_signal_560'] = self.fiber_data[red_col] / self.fiber_data[isos_col]
                    if not green_col:
                        self.fiber_data['normalized_signal'] = self.fiber_data['normalized_signal_560']
                        self.primary_signal = '560nm'
                
                self.channels = {
                    'time': time_col,
                    'isos': isos_col,
                    'green': green_col,
                    'red': red_col,
                    'events': events_col
                }
                
                messagebox.showinfo("Success", f"Fiber data loaded successfully\nPrimary signal: {self.primary_signal}")
                self.set_status(f"Fiber data loaded ({self.primary_signal})")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load fiber data: {str(e)}")
                self.set_status("Fiber data load failed")
                import traceback
                traceback.print_exc()
    
    def set_video_markers(self):
        """Set video start and end markers based on Input1 events in fiber data"""
        if self.fiber_data is None:
            messagebox.showwarning("No Data", "Please load fiber data first")
            return
        
        try:
            events_col = next((col for col in self.fiber_data.columns if 'events' in col.lower()), None)
            if events_col is None:
                messagebox.showerror("Error", "Events column not found in fiber data")
                return
            
            # Find Input1 events (assuming they mark video start/end)
            input1_events = self.fiber_data[self.fiber_data[events_col].str.contains('Input1', na=False)]
            
            if len(input1_events) < 2:
                messagebox.showerror("Error", "Could not find enough Input1 events (need at least 2)")
                return
            
            time_col = next((col for col in self.fiber_data.columns if 'timestamp' in col.lower()), None)
            self.video_start_fiber = input1_events[time_col].iloc[0]
            self.video_end_fiber = input1_events[time_col].iloc[-1]
            
            messagebox.showinfo("Success", 
                               f"Video markers set:\nStart: {self.video_start_fiber:.2f}s\nEnd: {self.video_end_fiber:.2f}s")
            self.set_status("Video markers set from fiber data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set video markers: {str(e)}")
            self.set_status("Video markers failed")
    
    def align_fiber_data(self):
        """Align fiber data with video markers"""
        if self.fiber_data is None or self.video_start_fiber is None or self.video_end_fiber is None:
            messagebox.showwarning("Missing Data", "Please load fiber data and set video markers first")
            return
        
        try:
            time_col = self.channels['time']
            
            # Crop fiber data to video period
            cropped_fiber = self.fiber_data[
                (self.fiber_data[time_col] >= self.video_start_fiber) & 
                (self.fiber_data[time_col] <= self.video_end_fiber)
            ].copy()
            
            # Reset time to start from zero
            cropped_fiber['aligned_time'] = cropped_fiber[time_col] - self.video_start_fiber
            
            self.processed_fiber_data = cropped_fiber
            self.set_status("Fiber data aligned to video timeline")
            
            # Plot aligned data
            self.plot_aligned_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to align fiber data: {str(e)}")
            self.set_status("Alignment failed")
    
    def plot_aligned_data(self):
        """Plot aligned fiber data"""
        if self.processed_fiber_data is None:
            messagebox.showwarning("No Data", "Please align data first")
            return
        
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        time_col = 'aligned_time'
        ax.plot(self.processed_fiber_data[time_col], self.processed_fiber_data['normalized_signal'], 
                'g-', label='Normalized Signal')
        
        # Add events if available
        events_col = self.channels['events']
        if events_col in self.processed_fiber_data.columns:
            event_data = self.processed_fiber_data[self.processed_fiber_data[events_col].notna()]
            for _, row in event_data.iterrows():
                ax.axvline(row[time_col], color='r', alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Signal')
        ax.set_title('Aligned Fiber Photometry Data')
        ax.legend()
        ax.grid(False)
        
        # Update plot frame based on analysis mode
        if self.analysis_mode == "freezing+fiber":
            self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
        elif self.analysis_mode == "pupil+fiber":
            self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
    
    def plot_fiber_data(self):
        """Plot raw fiber photometry data"""
        if self.fiber_data is None:
            messagebox.showwarning("No Data", "Please load fiber data first")
            return
        
        fig = Figure(figsize=(10, 8), dpi=100)
        
        time_col = self.channels['time']
        green_col = self.channels['green']
        red_col = self.channels['red']
        isos_col = self.channels['isos']
        
        # Determine number of subplots needed
        num_plots = 2  # Always show normalized signal + at least one raw signal
        if green_col and red_col:
            num_plots = 4
        elif green_col or red_col:
            num_plots = 3
        
        axes = []
        for i in range(num_plots):
            if i == 0:
                axes.append(fig.add_subplot(num_plots, 1, i+1))
            else:
                axes.append(fig.add_subplot(num_plots, 1, i+1, sharex=axes[0]))
        
        plot_idx = 0
        
        # Plot available raw signals
        if green_col:
            axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data[green_col], 
                              'g-', label='470nm Signal')
            axes[plot_idx].set_ylabel('470nm')
            axes[plot_idx].legend(loc='upper right')
            plot_idx += 1
            
        if red_col:
            axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data[red_col], 
                              'r-', label='560nm Signal')
            axes[plot_idx].set_ylabel('560nm')
            axes[plot_idx].legend(loc='upper right')
            plot_idx += 1
            
        # Always plot isosbestic reference
        axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data[isos_col], 
                          'b-', label='410/415nm Reference')
        axes[plot_idx].set_ylabel('Reference')
        axes[plot_idx].legend(loc='upper right')
        plot_idx += 1
        
        # Plot normalized signals
        if 'normalized_signal_470' in self.fiber_data.columns and 'normalized_signal_560' in self.fiber_data.columns:
            axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_470'], 
                              'm-', label='470/410')
            axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_560'], 
                              'c-', label='560/410')
            axes[plot_idx].set_ylabel('Normalized')
            axes[plot_idx].legend(loc='upper right')
        elif 'normalized_signal_470' in self.fiber_data.columns:
            axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_470'], 
                              'm-', label='470/410')
            axes[plot_idx].set_ylabel('Normalized')
            axes[plot_idx].legend(loc='upper right')
        elif 'normalized_signal_560' in self.fiber_data.columns:
            axes[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_560'], 
                              'c-', label='560/410')
            axes[plot_idx].set_ylabel('Normalized')
            axes[plot_idx].legend(loc='upper right')
        
        axes[-1].set_xlabel('Time (s)')
        
        # Highlight video period if markers are set
        if self.video_start_fiber is not None and self.video_end_fiber is not None:
            for ax in axes:
                ax.axvspan(self.video_start_fiber, self.video_end_fiber, color='yellow', alpha=0.3)
                ax.axvline(self.video_start_fiber, color='k', linestyle='--')
                ax.axvline(self.video_end_fiber, color='k', linestyle='--')
        
        fig.suptitle(f'Fiber Photometry Data ({self.primary_signal} primary)')
        fig.tight_layout()
        
        # Update plot frame based on analysis mode
        if self.analysis_mode == "freezing+fiber":
            self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
        elif self.analysis_mode == "pupil+fiber":
            self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
    
    # Preprocessing methods
    def apply_smoothing(self):
        """Apply smoothing to the fiber data"""
        if self.processed_fiber_data is None:
            messagebox.showwarning("No Data", "Please load and align fiber data first")
            return
        
        try:
            window_size = float(self.smooth_window_var.get())
            time_col = 'aligned_time'
            signal_col = 'normalized_signal'
            
            # Calculate sampling rate
            time_diff = np.diff(self.processed_fiber_data[time_col])
            avg_sampling_rate = 1 / np.mean(time_diff)
            
            # Convert window size in seconds to number of samples
            window_samples = int(window_size * avg_sampling_rate)
            if window_samples % 2 == 0:
                window_samples += 1  # Ensure odd window size
            
            # Apply Savitzky-Golay filter
            smoothed_signal = signal.savgol_filter(
                self.processed_fiber_data[signal_col], 
                window_length=window_samples, 
                polyorder=3
            )
            
            # Add smoothed signal to dataframe
            self.processed_fiber_data['smoothed_signal'] = smoothed_signal
            self.smoothed = True
            
            # Plot smoothed signal
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.plot(self.processed_fiber_data[time_col], self.processed_fiber_data[signal_col], 
                    'b-', alpha=0.5, label='Original')
            ax.plot(self.processed_fiber_data[time_col], smoothed_signal, 
                    'r-', label=f'Smoothed (window={window_size}s)')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Signal')
            ax.set_title('Smoothed Fiber Photometry Data')
            ax.legend()
            ax.grid(False)
            
            # Update plot
            if self.analysis_mode == "freezing+fiber":
                self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
            elif self.analysis_mode == "pupil+fiber":
                self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
            
            self.set_status(f"Applied smoothing with {window_size}s window")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply smoothing: {str(e)}")
            self.set_status("Smoothing failed")
    
    def apply_baseline_correction(self):
        """Apply baseline correction to the fiber data"""
        if self.processed_fiber_data is None:
            messagebox.showwarning("No Data", "Please load and align fiber data first")
            return
        
        method = self.baseline_method.get()
        if method == "none":
            return
        
        try:
            time_col = 'aligned_time'
            signal_col = 'smoothed_signal' if self.smoothed else 'normalized_signal'
            
            # Get signal to correct
            signal_data = self.processed_fiber_data[signal_col].values
            time_data = self.processed_fiber_data[time_col].values
            
            # Fit baseline
            if method == "exp":
                # Exponential decay fit
                def exp_decay(t, a, b, c):
                    return a * np.exp(-b * t) + c
                
                # Initial parameter guesses
                p0 = [signal_data.max() - signal_data.min(), 0.1, signal_data.min()]
                popt, _ = curve_fit(exp_decay, time_data, signal_data, p0=p0)
                baseline = exp_decay(time_data, *popt)
            
            elif method == "linear":
                # Linear fit
                coeffs = np.polyfit(time_data, signal_data, 1)
                baseline = np.polyval(coeffs, time_data)
            
            elif method == "poly":
                # Polynomial fit
                degree = int(self.poly_degree_var.get())
                coeffs = np.polyfit(time_data, signal_data, degree)
                baseline = np.polyval(coeffs, time_data)
            
            # Apply baseline correction
            corrected_signal = signal_data - baseline + np.median(signal_data)
            
            # Store results
            self.processed_fiber_data['baseline'] = baseline
            self.processed_fiber_data['baseline_corrected'] = corrected_signal
            self.baseline_corrected = True
            self.baseline_data = baseline
            
            # Plot baseline correction
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.plot(time_data, signal_data, 'b-', alpha=0.5, label='Original')
            ax.plot(time_data, baseline, 'r-', label='Baseline Fit')
            ax.plot(time_data, corrected_signal, 'g-', label='Baseline Corrected')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal')
            ax.set_title(f'Baseline Correction ({method})')
            ax.legend()
            ax.grid(False)
            
            # Update plot
            if self.analysis_mode == "freezing+fiber":
                self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
            elif self.analysis_mode == "pupil+fiber":
                self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
            
            self.set_status(f"Applied {method} baseline correction")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply baseline correction: {str(e)}")
            self.set_status("Baseline correction failed")
    
    def apply_motion_correction(self):
        """Apply motion correction to the fiber data"""
        if self.processed_fiber_data is None:
            messagebox.showwarning("No Data", "Please load and align fiber data first")
            return
        
        method = self.motion_method.get()
        if method == "none":
            return
        
        try:
            time_col = 'aligned_time'
            signal_col = 'baseline_corrected' if self.baseline_corrected else 'smoothed_signal' if self.smoothed else 'normalized_signal'
            
            signal_data = self.processed_fiber_data[signal_col].values
            time_data = self.processed_fiber_data[time_col].values
            
            if method == "isos":
                # Use isosbestic reference for motion correction
                if 'isos' not in self.channels or self.channels['isos'] not in self.processed_fiber_data.columns:
                    messagebox.showerror("Error", "Isosbestic reference not available for motion correction")
                    return
                
                isos_data = self.processed_fiber_data[self.channels['isos']].values
                
                # Fit linear regression: signal = a * isos + b
                coeffs = np.polyfit(isos_data, signal_data, 1)
                predicted = np.polyval(coeffs, isos_data)
                
                # Calculate corrected signal: residual + median
                corrected_signal = signal_data - predicted + np.median(signal_data)
                
            elif method == "baseline":
                # Use baseline period for motion correction
                start_time = float(self.baseline_start_var.get())
                end_time = float(self.baseline_end_var.get())
                
                # Extract baseline period
                baseline_mask = (time_data >= start_time) & (time_data <= end_time)
                if not np.any(baseline_mask):
                    messagebox.showerror("Error", "No data in the specified baseline period")
                    return
                
                baseline_signal = signal_data[baseline_mask]
                baseline_time = time_data[baseline_mask]
                
                # Calculate mean and std of baseline
                mean_baseline = np.mean(baseline_signal)
                std_baseline = np.std(baseline_signal)
                
                # Correct entire signal
                corrected_signal = (signal_data - mean_baseline) / std_baseline
            
            # Store results
            self.processed_fiber_data['motion_corrected'] = corrected_signal
            self.motion_corrected = True
            
            # Plot motion correction
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.plot(time_data, signal_data, 'b-', alpha=0.5, label='Original')
            ax.plot(time_data, corrected_signal, 'r-', label='Motion Corrected')
            
            if method == "baseline":
                ax.axvspan(start_time, end_time, color='g', alpha=0.2, label='Baseline Period')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal')
            ax.set_title(f'Motion Correction ({method})')
            ax.legend()
            ax.grid(False)
            
            # Update plot
            if self.analysis_mode == "freezing+fiber":
                self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
            elif self.analysis_mode == "pupil+fiber":
                self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
            
            self.set_status(f"Applied {method} motion correction")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply motion correction: {str(e)}")
            self.set_status("Motion correction failed")
    
    # ΔF/F and Z-score calculation
    def calculate_dff(self):
        """Calculate ΔF/F or Z-score for the fiber data"""
        if self.processed_fiber_data is None:
            messagebox.showwarning("No Data", "Please load and process fiber data first")
            return
        
        try:
            time_col = 'aligned_time'
            signal_col = 'motion_corrected' if self.motion_corrected else 'baseline_corrected' if self.baseline_corrected else 'smoothed_signal' if self.smoothed else 'normalized_signal'
            
            signal_data = self.processed_fiber_data[signal_col].values
            time_data = self.processed_fiber_data[time_col].values
            
            # Determine F0 calculation window
            window_size = float(self.f0_window_var.get())
            
            # Calculate F0 as rolling median
            # Calculate sampling rate
            time_diff = np.diff(time_data)
            avg_sampling_rate = 1 / np.mean(time_diff)
            window_samples = int(window_size * avg_sampling_rate)
            
            # Create rolling median
            f0 = pd.Series(signal_data).rolling(window=window_samples, min_periods=1, center=True).median().values
            
            # Calculate ΔF/F
            dff = (signal_data - f0) / f0
            
            # Store results
            self.processed_fiber_data['f0'] = f0
            self.processed_fiber_data['dff'] = dff
            
            # Calculate Z-score if selected
            if self.dff_method.get() == "zscore":
                zscore = (dff - np.mean(dff)) / np.std(dff)
                self.processed_fiber_data['zscore'] = zscore
                self.zscore_data = zscore
            
            self.dff_data = dff
            
            self.set_status("Calculated ΔF/F")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate ΔF/F: {str(e)}")
            self.set_status("ΔF/F calculation failed")
    
    def plot_dff(self):
        """Plot ΔF/F or Z-score"""
        if self.dff_data is None:
            messagebox.showwarning("No Data", "Please calculate ΔF/F first")
            return
        
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        time_col = 'aligned_time'
        plot_data = self.processed_fiber_data['zscore'] if self.dff_method.get() == "zscore" else self.processed_fiber_data['dff']
        plot_label = 'Z-score' if self.dff_method.get() == "zscore" else 'ΔF/F'
        
        ax.plot(self.processed_fiber_data[time_col], plot_data, 'b-', label=plot_label)
        
        # Add events if available
        events_col = self.channels['events']
        if events_col in self.processed_fiber_data.columns:
            event_data = self.processed_fiber_data[self.processed_fiber_data[events_col].notna()]
            for _, row in event_data.iterrows():
                ax.axvline(row[time_col], color='r', alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(plot_label)
        ax.set_title(f'Fiber Photometry {plot_label}')
        ax.legend()
        ax.grid(False)
        
        # Update plot
        if self.analysis_mode == "freezing+fiber":
            self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
        elif self.analysis_mode == "pupil+fiber":
            self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
    
    # Event analysis methods
    def plot_event_responses(self):
        """Plot event-aligned responses"""
        if self.dff_data is None or self.event_data is None:
            messagebox.showwarning("No Data", "Please calculate ΔF/F and load events first")
            return
        
        try:
            pre_event = float(self.pre_event_var.get())
            post_event = float(self.post_event_var.get())
            
            time_col = 'aligned_time'
            events_col = self.channels['events']
            
            # Get unique event types
            event_types = self.event_data['Event Type'].unique()
            
            # Create figure
            fig = Figure(figsize=(10, 8), dpi=100)
            
            # Create one subplot per event type
            axes = []
            for i, event_type in enumerate(event_types):
                axes.append(fig.add_subplot(len(event_types), 1, i+1))
                
                # Get events of this type
                events = self.event_data[self.event_data['Event Type'] == event_type]
                
                # Plot each event
                for j, (_, event) in enumerate(events.iterrows()):
                    event_time = event['start_time']
                    
                    # Extract peri-event data
                    mask = (self.processed_fiber_data[time_col] >= event_time - pre_event) & \
                           (self.processed_fiber_data[time_col] <= event_time + post_event)
                    peri_data = self.processed_fiber_data.loc[mask].copy()
                    
                    # Create relative time
                    peri_data['rel_time'] = peri_data[time_col] - event_time
                    
                    # Plot
                    axes[i].plot(peri_data['rel_time'], peri_data['dff'], 
                                alpha=0.5, label=f'Event {j+1}' if i == 0 else "")
                
                # Add event line
                axes[i].axvline(0, color='r', linestyle='--')
                
                axes[i].set_xlabel('Time relative to event (s)')
                axes[i].set_ylabel('ΔF/F')
                axes[i].set_title(f'Event Type: {event_type}')
                axes[i].grid(False)
            
            # Add common legend only in first subplot
            if len(events) > 0:
                axes[0].legend()
            
            fig.tight_layout()
            
            # Update plot
            if self.analysis_mode == "freezing+fiber":
                self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
            elif self.analysis_mode == "pupil+fiber":
                self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
            
            self.set_status(f"Plotted event responses ({len(events)} events)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot event responses: {str(e)}")
            self.set_status("Event response plot failed")
    
    def plot_event_heatmap(self):
        """Plot event-aligned heatmap"""
        if self.dff_data is None or self.event_data is None:
            messagebox.showwarning("No Data", "Please calculate ΔF/F and load events first")
            return
        
        try:
            pre_event = float(self.pre_event_var.get())
            post_event = float(self.post_event_var.get())
            
            time_col = 'aligned_time'
            events_col = self.channels['events']
            
            # Get unique event types
            event_types = self.event_data['Event Type'].unique()
            
            # Create figure
            fig = Figure(figsize=(10, 8), dpi=100)
            
            for i, event_type in enumerate(event_types):
                ax = fig.add_subplot(len(event_types), 1, i+1)
                
                # Get events of this type
                events = self.event_data[self.event_data['Event Type'] == event_type]
                
                # Create time vector for heatmap
                time_vector = np.linspace(-pre_event, post_event, 100)
                
                # Create matrix to store event-aligned signals
                event_matrix = np.zeros((len(events), len(time_vector)))
                
                # Interpolate each event's signal to common time vector
                for j, (_, event) in enumerate(events.iterrows()):
                    event_time = event['start_time']
                    
                    # Extract peri-event data
                    mask = (self.processed_fiber_data[time_col] >= event_time - pre_event) & \
                           (self.processed_fiber_data[time_col] <= event_time + post_event)
                    peri_data = self.processed_fiber_data.loc[mask].copy()
                    
                    # Create relative time
                    peri_data['rel_time'] = peri_data[time_col] - event_time
                    
                    # Interpolate to common time vector
                    interp = interp1d(peri_data['rel_time'], peri_data['dff'], 
                                     bounds_error=False, fill_value=np.nan)
                    event_matrix[j, :] = interp(time_vector)
                
                # Plot heatmap
                sns.heatmap(event_matrix, ax=ax, cmap='viridis', cbar=True, 
                           xticklabels=np.round(time_vector[::10], 1),
                           yticklabels=range(1, len(events)+1))
                
                # Add event line
                event_idx = np.where(time_vector >= 0)[0][0]
                ax.axvline(event_idx, color='r', linestyle='--')
                
                ax.set_xlabel('Time relative to event (s)')
                ax.set_ylabel('Event #')
                ax.set_title(f'Event Type: {event_type}')
            
            fig.tight_layout()
            
            # Update plot
            if self.analysis_mode == "freezing+fiber":
                self.create_plot_canvas(self.freezing_fiber_plot_frame, fig)
            elif self.analysis_mode == "pupil+fiber":
                self.create_plot_canvas(self.pupil_fiber_plot_frame, fig)
            
            self.set_status(f"Plotted event heatmap ({len(events)} events)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot event heatmap: {str(e)}")
            self.set_status("Event heatmap plot failed")
    
    def export_event_data(self):
        """Export event-aligned data to CSV"""
        if self.dff_data is None or self.event_data is None:
            messagebox.showwarning("No Data", "Please calculate ΔF/F and load events first")
            return
        
        try:
            pre_event = float(self.pre_event_var.get())
            post_event = float(self.post_event_var.get())
            
            time_col = 'aligned_time'
            
            # Create output data structure
            output_data = []
            
            # Process each event
            for i, (_, event) in enumerate(self.event_data.iterrows()):
                event_time = event['start_time']
                event_type = event['Event Type']
                
                # Extract peri-event data
                mask = (self.processed_fiber_data[time_col] >= event_time - pre_event) & \
                       (self.processed_fiber_data[time_col] <= event_time + post_event)
                peri_data = self.processed_fiber_data.loc[mask].copy()
                
                # Create relative time
                peri_data['rel_time'] = peri_data[time_col] - event_time
                
                # Add event metadata
                peri_data['event_id'] = i+1
                peri_data['event_type'] = event_type
                
                output_data.append(peri_data)
            
            # Combine all events
            combined_data = pd.concat(output_data)
            
            # Save to file
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv", 
                filetypes=[("CSV files", "*.csv")]
            )
            
            if save_path:
                combined_data.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Event data saved to:\n{save_path}")
                self.set_status(f"Exported event data to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export event data: {str(e)}")
            self.set_status("Event export failed")
    
    # Existing analysis methods from the original code
    def load_eztrack_results(self):
        """Load ezTrack (FreezeAnalysis) results"""
        self.eztrack_results_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.eztrack_results_path:
            filename = os.path.basename(self.eztrack_results_path)
            messagebox.showinfo("ezTrack Loaded", f"Selected ezTrack file:\n{filename}")
            self.set_status(f"ezTrack results loaded: {filename}")
    
    def load_events(self):
        """Load event CSV file (sound and shock timings)"""
        self.events_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.events_path:
            try:
                self.event_data = pd.read_csv(self.events_path)
                if self.experiment_start is not None:
                    self.event_data['start_absolute'] = self.experiment_start + self.event_data['start_time']
                    self.event_data['end_absolute'] = self.experiment_start + self.event_data['end_time']
                messagebox.showinfo("Events Loaded", "Event data loaded successfully")
                self.set_status("Event file loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load events file: {str(e)}")
                self.events_path = ""
    
    def load_timestamps(self):
        """Load timestamp file and extract experiment start time"""
        self.timestamp_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.timestamp_path:
            try:
                timestamps = pd.read_csv(self.timestamp_path)
                exp_start = timestamps[(timestamps['Device'] == 'Experiment') & 
                                      (timestamps['Action'] == 'Start')]['Timestamp'].values
                if len(exp_start) > 0:
                    self.experiment_start = exp_start[0]
                    self.set_status("Timestamp file loaded with experiment start time")
                messagebox.showinfo("Timestamps Loaded", f"Selected timestamp file:\n{self.timestamp_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse timestamp file: {str(e)}")
                self.timestamp_path = ""
    
    def analyze(self):
        if not all([self.eztrack_results_path, self.events_path]):
            messagebox.showwarning("Missing Inputs", "Please load ezTrack results and events file first.")
            return
        
        self.set_status("Running freezing analysis...")
        try:
            self.eztrack_result = pd.read_csv(self.eztrack_results_path)
            
            self.event_data = pd.read_csv(self.events_path)
            
            self.freezing_data = self.compute_freezing_from_eztrack(fps=30)
            
            self.compute_freezing_by_window(window_size_sec=30, fps=30)
            
            messagebox.showinfo("Done", "Freezing analysis completed!")
            self.set_status("Freezing analysis completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.set_status("Freezing analysis failed.")
    
    def compute_freezing_from_eztrack(self, fps):
        freezing_percentages = {}
        
        for idx, event in self.event_data.iterrows():
            event_start = event['start_time']*fps
            event_end = event['end_time']*fps
            
            event_frames = self.eztrack_result[
                (self.eztrack_result['Frame'] >= event_start) & 
                (self.eztrack_result['Frame'] <= event_end)]
            
            if len(event_frames) > 0:
                freezing_frames = event_frames[event_frames['Freezing'] == 100]
                freezing_percentage = len(freezing_frames) / len(event_frames) * 100
            else:
                freezing_percentage = 0
                
            freezing_percentages[f"Event_{idx+1}"] = round(freezing_percentage, 2)
        
        return freezing_percentages
    
    def compute_freezing_by_window(self, window_size_sec, fps):
        window_size_frames = window_size_sec * fps
        total_frames = len(self.eztrack_result)
        
        num_windows = math.ceil(total_frames / window_size_frames)
        
        freeze_sec = []
        window_times = []
        
        for i in range(num_windows):
            start_frame = i * window_size_frames
            end_frame = (i + 1) * window_size_frames - 1
            
            window_frames = self.eztrack_result[
                (self.eztrack_result['Frame'] >= start_frame) & 
                (self.eztrack_result['Frame'] <= end_frame)]
            
            if len(window_frames) > 0:
                freezing_frames = window_frames[window_frames['Freezing'] == 100]
                freezing_seconds = len(freezing_frames) / fps
            else:
                freezing_seconds = 0
                
            freeze_sec.append(freezing_seconds)
            
            window_center = (start_frame + end_frame) / (2 * fps)
            window_times.append(window_center)
        
        self.window_freeze = pd.DataFrame({
            'time': window_times,
            'freeze_sec': freeze_sec
        })
    
    def plot_freezing_windows(self):
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run the analysis first.")
            return
        if not self.events_path:
            messagebox.showwarning("Missing Events", "Please load events CSV first.")
            return
        
        try:
            self.set_status("Plotting freezing timeline...")
            
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            for _, row in self.event_data.iterrows():
                color = "#b6b6b6ff" if row['Event Type'] == 0 else "#fcb500f0"
                ax.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)
            
            x = self.window_freeze['time'].values
            y = self.window_freeze['freeze_sec'].values
            
            ax.plot(x, y, marker='o', color='black', label='Freezing')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Freezing Duration (s) per 30s window")
            ax.set_title("Freezing Timeline with Events")
            ax.grid(False)
            ax.legend()
            fig.tight_layout()
            
            # Update plot
            self.create_plot_canvas(self.freezing_plot_frame, fig)
            self.set_status("Freezing timeline plotted.")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot freezing timeline: {str(e)}")
            self.set_status("Plot failed.")
    
    def export_freezing_csv(self):
        if not self.freezing_data:
            messagebox.showwarning("No Data", "Please run the freezing analysis first.")
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            title="Export Freezing Results",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not save_path:
            return
        
        results = []
        for idx, event in self.event_data.iterrows():
            event_name = f"Event_{idx+1}"
            results.append({
                'Event': f"Event {idx+1}",
                'Start Time': event['start_time'],
                'End Time': event['end_time'],
                'Event Type': "Sound" if event['Event Type'] == 0 else "Shock",
                'Freezing Percentage': self.freezing_data.get(event_name, 0)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Exported", f"Freezing results saved to:\n{save_path}")
        self.set_status("Freezing results exported.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FearConditioningApp(root)
    root.mainloop()