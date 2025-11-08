import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import struct
import re
import os
import math
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
import fnmatch
from scipy.optimize import curve_fit

invert = True

class ModeSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Conditioning Analyzer")
        self.root.geometry("500x320")
        self.root.resizable(False, False)
        
        self.root.configure(bg="#f0f0f0")
        
        main_frame = ttk.Frame(root, padding=40)
        main_frame.pack(expand=True, fill="both")
        
        title_label = ttk.Label(main_frame, text="Fear Conditioning Analyzer", 
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=20)
        
        subtitle_label = ttk.Label(main_frame, text="Please Select Your Analysis Mode!", 
                                 font=("Arial", 12))
        subtitle_label.pack(pady=10)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        freezing_btn = ttk.Button(button_frame, text="Freezing Analysis", 
                                 command=lambda: self.select_mode("freezing"),
                                 width=20)
        freezing_btn.grid(row=0, column=0, padx=10, pady=10)
        
        pupil_btn = ttk.Button(button_frame, text="Pupil Analysis", 
                              command=lambda: self.select_mode("pupil"),
                              width=20)
        pupil_btn.grid(row=0, column=1, padx=10, pady=10)
        
        version_label = ttk.Label(main_frame, text="Version 2.0 | © 2025 ML Lab", 
                                 font=("Arial", 8))
        version_label.pack(side="bottom", pady=5)
        
    def select_mode(self, mode):
        response = messagebox.askyesno("Fiber Photometry", 
                                      "Do you want to include fiber photometry analysis?")
        
        self.root.destroy()
        
        root = tk.Tk()
        root.geometry("1200x950")
        root.title("Fear Conditioning Analyzer")
        
        if mode == "freezing":
            app = FreezingAnalyzerApp(root, include_fiber=response)
        else:
            app = PupilAnalyzerApp(root, include_fiber=response)
            
        root.mainloop()


class BaseAnalyzerApp:
    def __init__(self, root, include_fiber=False):
        self.root = root
        self.include_fiber = include_fiber
        self.fiber_data = None
        self.video_start_fiber = None
        self.video_end_fiber = None
        self.channels = {}
        self.primary_signal = None
        self.preprocessed_data = None
        self.dff_data = None
        self.zscore_data = None
        self.event_data = None
        self.fiber_cropped = None
        self.baseline_period = [0, 30]
        self.event_time_absolute = None
        self.channels = {}
        self.channel_data = {}
        self.active_channels = []
        self.preprocessed_data = None
        self.dff_data = None
        self.zscore_data = None
        self.smooth_window = tk.IntVar(value=11)
        self.smooth_order = tk.IntVar(value=1)
        self.baseline_start = tk.DoubleVar(value=0)
        self.baseline_end = tk.DoubleVar(value=30)
        self.apply_smooth = tk.BooleanVar(value=False)
        self.apply_baseline = tk.BooleanVar(value=False)
        self.apply_motion = tk.BooleanVar(value=False)
        self.target_signal_var = tk.StringVar(value="470")
        self.reference_signal_var = tk.StringVar(value="410")
        self.baseline_period = [0, 120]
        self.baseline_model = tk.StringVar(value="Polynomial")  # New: baseline model selection
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)
        
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10, ipadx=5, ipady=5)
        
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=False)
        
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        
        self.create_ui()
        
    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def create_ui(self):
        for widget in self.control_frame.winfo_children():
            widget.destroy()
            
        if self.include_fiber:
            self.create_step_navigation()
        
        self.create_main_controls()
    
    def create_step_navigation(self):
        step_frame = ttk.LabelFrame(self.control_frame, text="Analysis Steps")
        step_frame.pack(fill="x", padx=5, pady=5)
        
        steps = ttk.Frame(step_frame)
        steps.pack(fill="x")
        steps.columnconfigure(0, weight=1)
        steps.columnconfigure(1, weight=1)
        
        ttk.Button(steps, text="1. Data Load", command=self.show_step1).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(steps, text="2. Preprocessing", command=self.show_step2).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(steps, text="3. ΔF/F & Z-score", command=self.show_step3).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(steps, text="4. Event Analysis", command=self.show_step4).grid(row=1, column=1, padx=2, pady=2, sticky="ew")    
    
    def create_main_controls(self):
        pass

    def one_click_import(self):
        """One-click import: intelligently retrieve files based on experiment mode"""
        base_dir = filedialog.askdirectory(title="Select ear-tag-named experiment folder")
        if not base_dir:
            return
        
        try:
            self.set_status("Auto-scanning experiment files...")
            
            exp_mode = "freezing+fiber" if self.include_fiber else "freezing"
            if "PupilAnalyzerApp" in str(type(self)):
                exp_mode = "pupil+fiber" if self.include_fiber else "pupil"
            
            files_found   = {}
            imported_files = []
            
            if exp_mode == "freezing":
                required_files = {
                    'eztrack': ['freezingoutput.csv', '*freezing*.csv', '*eztrack*.csv'],
                    'events' : ['*events*.csv', '*timeline*.csv']
                }
            
            elif exp_mode == "freezing+fiber":
                required_files = {
                    'eztrack': ['freezingoutput.csv', '*freezing*.csv', '*eztrack*.csv'],
                    'events' : ['*events*.csv', '*timeline*.csv'],
                    'fiber'  : ['fluorescence.csv', '*fiber*.csv']
                }
            
            elif exp_mode == "pupil":
                required_files = {
                    'dlc'     : ['*dlc*.csv', '*deeplabcut*.csv'],
                    'events'  : ['*events*.csv', '*timeline*.csv'],
                    'timestamp': ['*timestamp*.csv', '*time*.csv'],
                    'ast2'    : ['*.ast2']
                }
            
            elif exp_mode == "pupil+fiber":
                required_files = {
                    'dlc'     : ['*dlc*.csv', '*deeplabcut*.csv'],
                    'events'  : ['*events*.csv', '*timeline*.csv'],
                    'timestamp': ['*timestamp*.csv', '*time*.csv'],
                    'ast2'    : ['*.ast2'],
                    'fiber'   : ['fluorescence.csv', '*fiber*.csv']
                }
            
            for file_type, patterns in required_files.items():
                found_file = None
                for root_path, dirs, files in os.walk(base_dir):
                    for file in files:
                        file_lower = file.lower()
                        for pattern in patterns:
                            if fnmatch.fnmatch(file_lower, pattern.lower()):
                                found_file = os.path.join(root_path, file)
                                files_found[file_type] = found_file
                                break
                        if found_file:
                            break
                    if found_file:
                        break
            
            if 'eztrack' in files_found:
                self.eztrack_results_path = files_found['eztrack']
                self.set_status(f"Loaded ezTrack: {os.path.basename(self.eztrack_results_path)}")
                imported_files.append("ezTrack results")
            
            if 'dlc' in files_found:
                self.dlc_results_path = files_found['dlc']
                filename = os.path.basename(self.dlc_results_path)
                match = re.search(r'cam(\d+)', filename)
                if match:
                    self.cam_id = int(match.group(1))
                self.set_status(f"Loaded DLC: Camera {self.cam_id}")
                imported_files.append("DLC results")
            
            if 'events' in files_found:
                self.events_path = files_found['events']
                try:
                    self.event_data = pd.read_csv(self.events_path)
                    first_time = self.event_data['start_time'].iloc[0]
                    self.event_time_absolute = first_time > 1e9
                    
                    if self.event_time_absolute:
                        self.event_data['start_absolute'] = self.event_data['start_time']
                        self.event_data['end_absolute'] = self.event_data['end_time']
                    imported_files.append("Events file")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load events: {str(e)}")
            
            if 'timestamp' in files_found:
                self.timestamp_path = files_found['timestamp']
                try:
                    timestamps = pd.read_csv(self.timestamp_path)
                    exp_start = timestamps[(timestamps['Device'] == 'Experiment') &
                                        (timestamps['Action'] == 'Start')]['Timestamp'].values
                    if len(exp_start) > 0:
                        self.experiment_start = exp_start[0]
                    imported_files.append("Timestamp file")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load timestamp: {str(e)}")
            
            if 'ast2' in files_found:
                self.ast2_path = files_found['ast2']
                try:
                    self.ast2_data = self.h_AST2_readData(self.ast2_path)
                    imported_files.append("AST2 file")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load AST2: {str(e)}")
            
            if 'fiber' in files_found:
                self.fiber_data_path = files_found['fiber']
                try:
                    self.load_fiber_data_silent()
                    imported_files.append("Fiber data")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load fiber: {str(e)}")
            
            if self.include_fiber and self.fiber_data is not None:
                if len(self.channel_data) > 1:
                    self.show_channel_selection_dialog()
                else:
                    self.active_channels = list(self.channel_data.keys())
                    self.set_status(f"Auto-selected channel: CH{self.active_channels[0]}")

            if exp_mode in ["freezing", "freezing+fiber"]:
                if self.eztrack_results_path and self.events_path:
                    self.analyze()
            elif exp_mode in ["pupil", "pupil+fiber"]:
                if self.dlc_results_path and self.timestamp_path:
                    self.run_pupil_analysis()
                if self.ast2_path:
                    self.align_ast2_data()
            
            if imported_files:
                files_list = "\n".join([f"✓ {f}" for f in imported_files])
                messagebox.showinfo("Import Complete",
                                    f"Mode: {exp_mode}\n\nSuccessfully imported:\n{files_list}")
            else:
                messagebox.showwarning("Warning", f"No files found for {exp_mode} mode")
        
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {str(e)}")
            self.set_status("Data import failed")

    def load_fiber_data_silent(self):
        self.fiber_data = pd.read_csv(self.fiber_data_path, skiprows=1, delimiter=',')
        self.fiber_data.columns = self.fiber_data.columns.str.strip()
        
        time_col = None
        possible_time_columns = ['timestamp', 'timems', 'time', 'time(ms)']
        for col in self.fiber_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in possible_time_columns):
                time_col = col
                break
        
        if not time_col:
            numeric_cols = self.fiber_data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                time_col = numeric_cols[0]
        
        self.fiber_data[time_col] = self.fiber_data[time_col] / 1000
        
        events_col = None
        for col in self.fiber_data.columns:
            if 'event' in col.lower():
                events_col = col
                break
        
        self.channels = {'time': time_col, 'events': events_col}
        
        self.channel_data = {}
        channel_pattern = re.compile(r'CH(\d+)-(\d+)', re.IGNORECASE)
        
        for col in self.fiber_data.columns:
            match = channel_pattern.match(col)
            if match:
                channel_num = int(match.group(1))
                wavelength = int(match.group(2))
                
                if channel_num not in self.channel_data:
                    self.channel_data[channel_num] = {'410': None, '415': None, '470': None, '560': None}
                
                if wavelength == 410 or wavelength == 415:
                    self.channel_data[channel_num]['410' if wavelength == 410 else '415'] = col
                elif wavelength == 470:
                    self.channel_data[channel_num]['470'] = col
                elif wavelength == 560:
                    self.channel_data[channel_num]['560'] = col
        
        self.set_status(f"Fiber data loaded, {len(self.channel_data)} channels detected")

    def load_fiber_data(self):
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
                
                if not time_col:
                    numeric_cols = self.fiber_data.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0 and self.fiber_data[numeric_cols[0]].dtype in [np.int64, np.float64]:
                        time_col = numeric_cols[0]
                        self.set_status(f"Using first numeric column as time: {time_col}")
                    else:
                        messagebox.showerror("Error", "Timestamp column not found in fiber data file")
                        self.fiber_data = None
                        return
                self.fiber_data[time_col] = self.fiber_data[time_col] / 1000

                events_col = None
                for col in self.fiber_data.columns:
                    if 'event' in col.lower():
                        events_col = col
                        break
                
                self.channels = {'time': time_col, 'events': events_col}
                self.channel_data = {}
                channel_pattern = re.compile(r'CH(\d+)-(\d+)', re.IGNORECASE)
                
                for col in self.fiber_data.columns:
                    match = channel_pattern.match(col)
                    if match:
                        channel_num = int(match.group(1))
                        wavelength = int(match.group(2))
                        
                        if channel_num not in self.channel_data:
                            self.channel_data[channel_num] = {
                                '410': None,
                                '415': None,
                                '470': None,
                                '560': None
                            }
                        
                        if wavelength == 410 or wavelength == 415:
                            self.channel_data[channel_num]['410' if wavelength == 410 else '415'] = col
                        elif wavelength == 470:
                            self.channel_data[channel_num]['470'] = col
                        elif wavelength == 560:
                            self.channel_data[channel_num]['560'] = col
                
                if len(self.channel_data) == 1:
                    self.active_channels = list(self.channel_data.keys())
                    self.set_status(f"Single channel detected: CH{self.active_channels[0]}")
                else:
                    self.show_channel_selection_dialog()
                
                self.set_status(f"Fiber data loaded with {len(self.channel_data)} channels")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load fiber data: {str(e)}")
                self.set_status("Fiber data load failed")
    
    def show_channel_selection_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Channels")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Select channels to analyze:", font=("Arial", 10, "bold")).pack(pady=5)
        
        channels_frame = ttk.Frame(frame)
        channels_frame.pack(fill="both", expand=True, pady=10)
        
        self.channel_vars = {}
        for channel_num in sorted(self.channel_data.keys()):
            var = tk.BooleanVar(value=(channel_num == 1))
            self.channel_vars[channel_num] = var
            
            chk = ttk.Checkbutton(
                channels_frame, 
                text=f"Channel {channel_num} (410/470: {self.channel_data[channel_num]['410'] is not None}, 560: {self.channel_data[channel_num]['560'] is not None})",
                variable=var
            )
            chk.pack(anchor="w", padx=20, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Select All", command=lambda: self.toggle_all_channels(True)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: self.toggle_all_channels(False)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Confirm", command=lambda: self.finalize_channel_selection(dialog)).pack(side="right", padx=5)
    
    def toggle_all_channels(self, select):
        for var in self.channel_vars.values():
            var.set(select)
    
    def finalize_channel_selection(self, dialog):
        self.active_channels = []
        for channel_num, var in self.channel_vars.items():
            if var.get():
                self.active_channels.append(channel_num)
        
        if not self.active_channels:
            messagebox.showwarning("No Channels Selected", "Please select at least one channel")
            return
        
        dialog.destroy()
        self.set_status(f"Selected channels: {', '.join([f'CH{ch}' for ch in self.active_channels])}")

        if self.include_fiber and self.fiber_data is not None:
            self.set_video_markers()
    
    def set_video_markers(self):
        if self.fiber_data is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load fiber data and select channels first")
            return
        
        try:
            events_col = self.channels.get('events')
            if events_col is None or events_col not in self.fiber_data.columns:
                messagebox.showerror("Error", "Events column not found in fiber data")
                return
            
            input1_events = self.fiber_data[self.fiber_data[events_col].str.contains('Input1', na=False)]
            
            if len(input1_events) < 2:
                messagebox.showerror("Error", "Could not find enough Input1 events (need at least 2)")
                return
            
            time_col = self.channels['time']
            self.video_start_fiber = input1_events[time_col].iloc[0]
            self.video_end_fiber = input1_events[time_col].iloc[-1]
            
            self.fiber_cropped = self.fiber_data[
                (self.fiber_data[time_col] >= self.video_start_fiber) & 
                (self.fiber_data[time_col] <= self.video_end_fiber)].copy()
            
            self.preprocessed_data = None
            self.dff_data = None
            self.zscore_data = None
            
            if hasattr(self, 'event_data') and not self.event_time_absolute:
                self.convert_relative_to_absolute()
            
            messagebox.showinfo("Success", 
                               f"Video markers set:\nStart: {self.video_start_fiber:.2f}s\nEnd: {self.video_end_fiber:.2f}s")
            self.set_status("Video markers set from fiber data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set video markers: {str(e)}")
            self.set_status("Video markers failed")
    
    def plot_raw_data(self):
        if self.fiber_cropped is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            time_col = self.channels['time']
            time_data = self.fiber_cropped[time_col] - self.video_start_fiber
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    for wavelength, col_name in self.channel_data[channel_num].items():
                        if col_name and col_name in self.fiber_cropped.columns:
                            color = colors[i % len(colors)]
                            linestyle = '-' if wavelength == self.target_signal_var.get() else '--'
                            alpha = 1.0 if wavelength == self.target_signal_var.get() else 0.7
                            ax.plot(time_data, self.fiber_cropped[col_name], 
                                   f'{color}{linestyle}', linewidth=1, alpha=alpha, 
                                   label=f'CH{channel_num} {wavelength}nm')
            
            ax.set_title("Raw Fiber Photometry Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.legend()
            self.canvas.draw()
            self.set_status("Raw fiber data plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            self.set_status("Raw plot failed")
    
    def smooth_data(self):
        if self.fiber_cropped is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            window_size = simpledialog.askinteger("Smoothing", 
                                                "Enter smoothing window size (odd number):", 
                                                initialvalue=11)
            if window_size is None:
                return
                
            if window_size % 2 == 0:
                window_size += 1
                
            poly_order = simpledialog.askinteger("Smoothing", 
                                               "Enter polynomial order:", 
                                               initialvalue=1)
            if poly_order is None:
                return
                
            if self.preprocessed_data is None:
                self.preprocessed_data = self.fiber_cropped.copy()
            
            for channel_num in self.active_channels:
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if target_col and target_col in self.preprocessed_data.columns:
                        smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                        self.preprocessed_data[smoothed_col] = savgol_filter(
                            self.preprocessed_data[target_col], window_size, poly_order)
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                    
                    if target_col and target_col in self.preprocessed_data.columns:
                        color = colors[i % len(colors)]
                        ax.plot(time_data, self.preprocessed_data[target_col], 
                               f'{color}-', alpha=0.5, label=f'CH{channel_num} Raw')
                    
                    if smoothed_col in self.preprocessed_data.columns:
                        color = colors[i % len(colors)]
                        ax.plot(time_data, self.preprocessed_data[smoothed_col], 
                               f'{color}-', linewidth=2, label=f'CH{channel_num} Smoothed')
            
            ax.set_title(f"Smoothed Fiber Data (window={window_size}, order={poly_order})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            ax.legend()
            self.canvas.draw()
            self.set_status(f"Data smoothed with window={window_size}, order={poly_order}")
        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed: {str(e)}")
            self.set_status("Smoothing failed")
    
    def baseline_correction(self):
        if self.fiber_cropped is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            model_type = simpledialog.askstring("Baseline Model", 
                                              "Select model type (Polynomial or Exponential):",
                                              initialvalue="Polynomial")
            if model_type is None:
                return
            
            if self.preprocessed_data is None:
                self.preprocessed_data = self.fiber_cropped.copy()
                
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            baseline_mask = (time_data >= 0)
            
            if not any(baseline_mask):
                messagebox.showerror("Error", "No data available")
                return
                
            for channel_num in self.active_channels:
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if not target_col or target_col not in self.preprocessed_data.columns:
                        continue
                    
                    smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                    if smoothed_col in self.preprocessed_data.columns:
                        signal_col = smoothed_col
                    else:
                        signal_col = target_col
                    
                    signal_data = self.preprocessed_data[signal_col]
                    
                    if model_type.lower() == "exponential":
                        def exp_model(x, a, b, c):
                            return a * np.exp(-b * x) + c
                        
                        x0 = time_data.values - time_data.min()
                        y0 = signal_data.values
                        
                        try:
                            params, _ = curve_fit(exp_model, x0, y0, p0=[max(y0), 0.1, min(y0)])
                            baseline_pred = exp_model(x0, *params)
                        except:
                            X = time_data.values.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, signal_data.values)
                            baseline_pred = model.predict(X)
                    else:
                        X = time_data.values.reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X, signal_data.values)
                        baseline_pred = model.predict(X)
                    
                    baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                    self.preprocessed_data[baseline_corrected_col] = signal_data - baseline_pred
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                    if baseline_corrected_col in self.preprocessed_data.columns:
                        color = colors[i % len(colors)]
                        ax.plot(time_data, self.preprocessed_data[baseline_corrected_col], 
                               f'{color}-', label=f'CH{channel_num} Baseline Corrected')
            
            ax.set_title(f"Baseline Correction ({model_type} Model)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            ax.legend()
            self.canvas.draw()
            self.set_status(f"Baseline correction applied ({model_type} model)")
        except Exception as e:
            messagebox.showerror("Error", f"Baseline correction failed: {str(e)}")
            self.set_status("Baseline correction failed")
    
    def motion_correction(self):
        if self.reference_signal_var.get() != "410":
            messagebox.showwarning("Invalid Reference", "Motion correction requires 410nm as reference signal")
            return
            
        if self.fiber_cropped is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            if self.preprocessed_data is None:
                self.preprocessed_data = self.fiber_cropped.copy()
                
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            for channel_num in self.active_channels:
                if channel_num in self.channel_data:
                    ref_col = self.channel_data[channel_num].get('410')
                    if not ref_col or ref_col not in self.preprocessed_data.columns:
                        self.set_status(f"No 410nm data for channel CH{channel_num}")
                        continue
                    
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if not target_col or target_col not in self.preprocessed_data.columns:
                        continue
                    
                    if f"CH{channel_num}_baseline_corrected" in self.preprocessed_data.columns:
                        signal_col = f"CH{channel_num}_baseline_corrected"
                    elif f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in self.preprocessed_data.columns:
                        signal_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                    else:
                        signal_col = target_col
                    
                    signal_data = self.preprocessed_data[signal_col]
                    ref_data = self.preprocessed_data[ref_col]
                    
                    X = ref_data.values.reshape(-1, 1)
                    y = signal_data.values
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    predicted_signal = model.predict(X)
                    
                    motion_corrected_col = f"CH{channel_num}_motion_corrected"
                    self.preprocessed_data[motion_corrected_col] = signal_data - predicted_signal
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    motion_corrected_col = f"CH{channel_num}_motion_corrected"
                    if motion_corrected_col in self.preprocessed_data.columns:
                        color = colors[i % len(colors)]
                        ax.plot(time_data, self.preprocessed_data[motion_corrected_col], 
                               f'{color}-', label=f'CH{channel_num} Motion Corrected')
            
            ax.set_title("Motion Correction")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            ax.legend()
            self.canvas.draw()
            self.set_status("Motion correction applied")
        except Exception as e:
            messagebox.showerror("Error", f"Motion correction failed: {str(e)}")
            self.set_status("Motion correction failed")
    
    def calculate_and_plot_dff(self):
        if self.preprocessed_data is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please preprocess data and select channels first")
            return
        
        try:
            if self.dff_data is None:
                self.dff_data = {}
            
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
            
            if not any(baseline_mask):
                messagebox.showerror("Error", "No data in baseline period")
                return
                
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    if f"CH{channel_num}_motion_corrected" in self.preprocessed_data.columns:
                        signal_col = f"CH{channel_num}_motion_corrected"
                    elif f"CH{channel_num}_baseline_corrected" in self.preprocessed_data.columns:
                        signal_col = f"CH{channel_num}_baseline_corrected"
                    elif f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in self.preprocessed_data.columns:
                        signal_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                    else:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if not target_col or target_col not in self.preprocessed_data.columns:
                            continue
                        signal_col = target_col
                    
                    signal_data = self.preprocessed_data[signal_col]
                    
                    if self.reference_signal_var.get() == "baseline":
                        baseline_data = signal_data[baseline_mask]
                        F0 = np.median(baseline_data)
                        dff_data = (signal_data - F0) / F0
                    elif self.reference_signal_var.get() == "410":
                        ref_col = self.channel_data[channel_num].get('410')
                        if not ref_col or ref_col not in self.preprocessed_data.columns:
                            continue
                        ref_data = self.preprocessed_data[ref_col]
                        dff_data = (signal_data - ref_data) / ref_data
                    else:
                        messagebox.showerror("Error", "Invalid reference signal selected")
                        return
                    
                    dff_col = f"CH{channel_num}_dff"
                    self.preprocessed_data[dff_col] = dff_data
                    self.dff_data[channel_num] = dff_data
                    
                    color = colors[i % len(colors)]
                    ax.plot(time_data, dff_data, f'{color}-', label=f'CH{channel_num} ΔF/F')
            
            ax.axvspan(self.baseline_period[0], self.baseline_period[1], color='green', alpha=0.2)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.set_title("ΔF/F")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔF/F")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
            self.set_status("ΔF/F calculated")
        except Exception as e:
            messagebox.showerror("Error", f"ΔF/F calculation failed: {str(e)}")
            self.set_status("ΔF/F calculation failed")
    
    def calculate_and_plot_zscore(self):
        try:
            if self.dff_data is None or not self.active_channels:
                messagebox.showwarning("No Data", "Please calculate ΔF/F first")
                return
                
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
            
            if not any(baseline_mask):
                messagebox.showerror("Error", "No data in baseline period")
                return
                
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.dff_data:
                    dff_data = self.dff_data[channel_num]
                    baseline_dff = dff_data[baseline_mask]
                    
                    if len(baseline_dff) < 2:
                        continue
                    
                    mean_dff = np.mean(baseline_dff)
                    std_dff = np.std(baseline_dff)
                    
                    zscore_data = (dff_data - mean_dff) / std_dff
                    
                    zscore_col = f"CH{channel_num}_zscore"
                    self.preprocessed_data[zscore_col] = zscore_data
                    self.zscore_data = zscore_data
                    
                    color = colors[i % len(colors)]
                    ax.plot(time_data, zscore_data, f'{color}-', label=f'CH{channel_num} Z-score')
            
            ax.axvspan(self.baseline_period[0], self.baseline_period[1], color='green', alpha=0.2)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.set_title("Z-score")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Z-score")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
            self.set_status("Z-score calculated")
        except Exception as e:
            messagebox.showerror("Error", f"Z-score calculation failed: {str(e)}")
            self.set_status("Z-score calculation failed")
    
    def plot_event_related_activity(self):
        if self.event_data is None:
            messagebox.showwarning("No Event Data", "Please load event data first")
            return
            
        if self.dff_data is None:
            self.calculate_and_plot_dff()
            
        if self.dff_data is None:
            return
            
        try:
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            selected_type = int(self.event_type.get())
            
            if pre_window is None or post_window is None or selected_type is None:
                return
            
            event_types = self.event_data['Event Type'].unique()
            if 0 in event_types:
                event_types = event_types[event_types != 0]

            if selected_type not in event_types:
                messagebox.showerror("Error", f"Event type {selected_type} not found")
                return
                
            events = self.event_data[self.event_data['Event Type'] == selected_type]
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            all_activities = {}
            
            for channel_num in self.active_channels:
                if channel_num in self.dff_data:
                    dff_data = self.dff_data[channel_num]
                    time_col = self.channels.get('time', self.preprocessed_data.columns[0])
                    time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                    
                    channel_activities = []
                    
                    for _, event in events.iterrows():
                        if self.event_time_absolute:
                            start_time = event['start_absolute'] - self.video_start_fiber
                            end_time = event['end_absolute'] - self.video_start_fiber
                        else:
                            start_time = event['start_time']
                            end_time = event['end_time']
                        
                        event_start = start_time - pre_window
                        event_end = end_time + post_window
                        
                        event_mask = (time_data >= event_start) & (time_data <= event_end)
                        event_time_relative = time_data[event_mask] - (start_time)
                        event_dff = dff_data[event_mask]
                        
                        channel_activities.append((event_time_relative, event_dff))
                    
                    all_activities[channel_num] = channel_activities
            
            # Plot all events
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in all_activities:
                    color = colors[i % len(colors)]
                    for event_time, event_dff in all_activities[channel_num]:
                        ax.plot(event_time, event_dff, f'{color}-', alpha=0.3)
            
            # Plot mean and SEM
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in all_activities and all_activities[channel_num]:
                    color = colors[i % len(colors)]
                    
                    min_time = min([min(et) for et, _ in all_activities[channel_num]])
                    max_time = max([max(et) for et, _ in all_activities[channel_num]])
                    all_times = np.linspace(min_time, max_time, 100)
                    
                    interpolated_dff = []
                    
                    for event_time, event_dff in all_activities[channel_num]:
                        if len(event_time) < 2:
                            continue
                        interp_dff = np.interp(all_times, event_time, event_dff)
                        interpolated_dff.append(interp_dff)
                    
                    if interpolated_dff:
                        mean_dff = np.mean(interpolated_dff, axis=0)
                        sem_dff = np.std(interpolated_dff, axis=0) / np.sqrt(len(interpolated_dff))
                        
                        ax.plot(all_times, mean_dff, f'{color}-', linewidth=2, label=f'CH{channel_num} Mean')
                        ax.fill_between(all_times, mean_dff - sem_dff, mean_dff + sem_dff, 
                                       color=color, alpha=0.2)
            
            # Mark event periods
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            if not events.empty:
                event_duration = events.iloc[0]['end_time'] - events.iloc[0]['start_time']
                ax.axvline(event_duration, color='k', linestyle='--', linewidth=1)
                ax.axvspan(0, event_duration, color='yellow', alpha=0.2)
                
                # Also mark type 2 events if they occur during type 1 events
                type2_events = self.event_data[self.event_data['Event Type'] == 2]
                for _, event2 in type2_events.iterrows():
                    if self.event_time_absolute:
                        start2 = event2['start_absolute'] - self.video_start_fiber
                        end2 = event2['end_absolute'] - self.video_start_fiber
                    else:
                        start2 = event2['start_time']
                        end2 = event2['end_time']
                    
                    # Check if type 2 event overlaps with any type 1 event window
                    for _, event1 in events.iterrows():
                        if self.event_time_absolute:
                            event1_start = event1['start_absolute'] - self.video_start_fiber
                            event1_end = event1['end_absolute'] - self.video_start_fiber
                        else:
                            event1_start = event1['start_time']
                            event1_end = event1['end_time']
                        
                        # If type 2 event occurs during type 1 event
                        if start2 >= event1_start and end2 <= event1_end:
                            rel_start = start2 - event1_start
                            ax.axvline(rel_start, color='k', linestyle='--', linewidth=1)
            
            ax.set_title(f"Event-Related Activity (Type: {selected_type})")
            ax.set_xlabel("Time Relative to Event Start (s)")
            ax.set_ylabel("ΔF/F")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
            self.set_status(f"Event-related activity plotted for type {selected_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Event analysis failed: {str(e)}")
            self.set_status("Event analysis failed")
    
    def plot_heatmap(self):
        if self.event_data is None:
            messagebox.showwarning("No Event Data", "Please load event data first")
            return
            
        if self.dff_data is None:
            self.calculate_and_plot_dff()
            
        if self.dff_data is None:
            return
            
        try:
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            selected_type = int(self.event_type.get())
            
            if pre_window is None or post_window is None or selected_type is None:
                return
            
            event_types = self.event_data['Event Type'].unique()
            if 0 in event_types:
                event_types = event_types[event_types != 0]
                
            if selected_type not in event_types:
                messagebox.showerror("Error", f"Event type {selected_type} not found")
                return
                
            events = self.event_data[self.event_data['Event Type'] == selected_type]
            
            num_channels = len(self.active_channels)
            if num_channels == 0:
                return
                
            self.fig.clear()
            
            for idx, channel_num in enumerate(self.active_channels):
                if channel_num not in self.dff_data:
                    continue
                    
                ax = self.fig.add_subplot(num_channels, 1, idx+1)
                
                dff_data = self.dff_data[channel_num]
                time_col = self.channels.get('time', self.preprocessed_data.columns[0])
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                all_activities = []
                
                for _, event in events.iterrows():
                    if self.event_time_absolute:
                        start_time = event['start_absolute'] - self.video_start_fiber
                        end_time = event['end_absolute'] - self.video_start_fiber
                    else:
                        start_time = event['start_time']
                        end_time = event['end_time']
                    
                    event_start = start_time - pre_window
                    event_end = end_time + post_window
                    
                    event_mask = (time_data >= event_start) & (time_data <= event_end)
                    event_time_relative = time_data[event_mask] - (start_time)
                    event_dff = dff_data[event_mask]
                    
                    interp_time = np.linspace(-pre_window, event_end - start_time, 100)
                    interp_dff = np.interp(interp_time, event_time_relative, event_dff)
                    
                    all_activities.append(interp_dff)
                
                if not all_activities:
                    continue
                    
                activity_matrix = np.array(all_activities)
                
                colors = ["blue", "white", "red"]
                cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors, N=256)
                
                event_duration = events.iloc[0]['end_time'] - events.iloc[0]['start_time']
                im = ax.imshow(activity_matrix, aspect='auto', cmap=cmap, 
                              extent=[-pre_window, event_duration + post_window, 
                                     0, len(events)])
                
                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                ax.axvline(event_duration, color='k', linestyle='--', linewidth=1)
                
                # Also mark type 2 events if they occur during type 1 events
                type2_events = self.event_data[self.event_data['Event Type'] == 2]
                for event_idx, (_, event1) in enumerate(events.iterrows()):
                    if self.event_time_absolute:
                        event1_start = event1['start_absolute'] - self.video_start_fiber
                        event1_end = event1['end_absolute'] - self.video_start_fiber
                    else:
                        event1_start = event1['start_time']
                        event1_end = event1['end_time']
                    
                    for _, event2 in type2_events.iterrows():
                        if self.event_time_absolute:
                            start2 = event2['start_absolute'] - self.video_start_fiber
                            end2 = event2['end_absolute'] - self.video_start_fiber
                        else:
                            start2 = event2['start_time']
                            end2 = event2['end_time']
                        
                        # If type 2 event occurs during type 1 event
                        if start2 >= event1_start and end2 <= event1_end:
                            rel_start = start2 - event1_start
                            ax.axvline(rel_start, color='k', linestyle='--', linewidth=1)
                
                ax.set_title(f"Channel CH{channel_num}")
                if idx == num_channels - 1:
                    ax.set_xlabel("Time Relative to Event Start (s)")
                ax.set_ylabel("Event #")
                ax.grid(False)
                
                self.fig.colorbar(im, ax=ax, label="ΔF/F")
            
            self.fig.suptitle(f"Event-Related Activity Heatmap (Type: {selected_type})")
            self.fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            self.canvas.draw()
            self.set_status(f"Heatmap plotted for type {selected_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Heatmap creation failed: {str(e)}")
            self.set_status("Heatmap failed")

    def load_events(self):
        """Load event CSV file, auto-detect and convert time format if needed"""
        self.events_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.events_path:
            try:
                self.event_data = pd.read_csv(self.events_path)
                
                first_time = self.event_data['start_time'].iloc[0]
                self.event_time_absolute = first_time > 1e9
                
                if self.event_time_absolute:
                    self.event_data['start_absolute'] = self.event_data['start_time']
                    self.event_data['end_absolute'] = self.event_data['end_time']
                    self.set_status("Loaded absolute time events")
                else:
                    self.set_status("Loaded relative time events")
                
                if not self.event_time_absolute and self.video_start_fiber is not None:
                    self.convert_relative_to_absolute()
                
                messagebox.showinfo("Events Loaded", "Event data loaded successfully")
                self.set_status("Event file loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load events file: {str(e)}")
                self.events_path = ""
    
    def convert_relative_to_absolute(self):
        """Convert relative event times to absolute times using video start time"""
        if not hasattr(self, 'video_start_fiber') or not self.video_start_fiber:
            return
            
        if self.event_time_absolute:
            return
            
        self.event_data['start_absolute'] = self.video_start_fiber + self.event_data['start_time']
        self.event_data['end_absolute'] = self.video_start_fiber + self.event_data['end_time']
        self.event_time_absolute = True
        self.set_status("Converted event times to absolute")

class FreezingAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("Freezing Analyzer")
        self.eztrack_results_path = ""
        self.freezing_data = {}
        self.window_freeze = None
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)
        
        freezing_frame = ttk.LabelFrame(control_panel, text="Freezing Analysis")
        freezing_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(freezing_frame, text="Load ezTrack Results", command=self.load_eztrack_Results).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(freezing_frame, text="Load Events CSV", command=self.load_events).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(freezing_frame, text="Run Freezing Analysis", command=self.analyze).grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(freezing_frame, text="Plot Freezing Timeline", command=self.plot_freezing_windows).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(freezing_frame, text="Export Results", command=self.export_freezing_csv).grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(freezing_frame, text="One-click Import", command=self.one_click_import).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(fiber_frame, text="Load Fiber Data", command=self.load_fiber_data).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Align Fiber Data", command=self.set_video_markers).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=1, column=0, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Plot Freezing w/Fiber", command=self.plot_freezing_with_fiber).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
            
    def show_step1(self):
        self.set_status("Step 1: Data Loading")
    
    def show_step2(self):
        self.set_status("Step 2: Preprocessing")
        
        self.step2_frame = ttk.LabelFrame(self.control_frame, text="Preprocessing")
        self.step2_frame.pack(fill="x", padx=5, pady=5)
        
        signal_frame = ttk.Frame(self.step2_frame)
        signal_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(signal_frame, text="Target Signal:").grid(row=0, column=0, sticky="w")
        target_options = ["470", "560"]
        target_menu = ttk.OptionMenu(signal_frame, self.target_signal_var, "470", *target_options)
        target_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        ttk.Label(signal_frame, text="Reference Signal:").grid(row=1, column=0, sticky="w")
        ref_options = ["410", "baseline"]
        ref_menu = ttk.OptionMenu(signal_frame, self.reference_signal_var, "410", *ref_options)
        ref_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        self.baseline_frame = ttk.Frame(self.step2_frame)
        self.baseline_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(self.baseline_frame, text="Baseline Period (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_start, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(self.baseline_frame, text="to").grid(row=0, column=2)
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_end, width=6).grid(row=0, column=3, padx=2)
        
        if self.reference_signal_var.get() != "baseline":
            self.baseline_frame.pack_forget()
        
        model_frame = ttk.Frame(self.step2_frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_frame, text="Baseline Model:").grid(row=0, column=0, sticky="w")
        model_options = ["Polynomial", "Exponential"]
        model_menu = ttk.OptionMenu(model_frame, self.baseline_model, "Polynomial", *model_options)
        model_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        smooth_frame = ttk.Frame(self.step2_frame)
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(smooth_frame, text="Apply Smoothing", variable=self.apply_smooth,
                       command=lambda: self.toggle_widgets(smooth_frame, self.apply_smooth.get(), 1)).grid(row=0, column=0, sticky="w")
        
        param_frame = ttk.Frame(smooth_frame)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Label(param_frame, text="Window Size:").grid(row=0, column=0, sticky="w")
        ttk.Scale(param_frame, from_=3, to=101, orient=tk.HORIZONTAL, 
                 length=150, variable=self.smooth_window,
                 command=lambda v: self.smooth_window.set(int(float(v)))).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_window).grid(row=0, column=2)
        
        ttk.Label(param_frame, text="Polynomial Order:").grid(row=1, column=0, sticky="w")
        ttk.Scale(param_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                 length=150, variable=self.smooth_order,
                 command=lambda v: self.smooth_order.set(int(float(v)))).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_order).grid(row=1, column=2)
        
        param_frame.grid_remove()
        
        baseline_corr_frame = ttk.Frame(self.step2_frame)
        baseline_corr_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=self.apply_baseline).grid(row=0, column=0, sticky="w")
        
        motion_frame = ttk.Frame(self.step2_frame)
        motion_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(motion_frame, text="Apply Motion Correction", variable=self.apply_motion,
                       command=lambda: self.toggle_motion_correction()).grid(row=0, column=0, sticky="w")
        
        ttk.Button(self.step2_frame, text="Apply Preprocessing and Plot", 
                  command=self.apply_preprocessing).pack(pady=10)
        
        self.reference_signal_var.trace_add("write", self.update_baseline_ui)
    
    def update_baseline_ui(self, *args):
        if self.reference_signal_var.get() == "baseline":
            self.baseline_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.baseline_frame.pack_forget()
    
    def toggle_motion_correction(self):
        if self.reference_signal_var.get() != "410":
            self.apply_motion.set(False)
            messagebox.showinfo("Info", "Motion correction requires 410nm as reference signal")
    
    def toggle_widgets(self, parent_frame, show, index):
        children = parent_frame.winfo_children()
        if len(children) > index:
            if show:
                children[index].grid()
            else:
                children[index].grid_remove()
    
    def apply_preprocessing(self):
        if self.fiber_cropped is None:
            messagebox.showwarning("No Data", "Please load fiber data first")
            return
            
        try:
            self.preprocessed_data = self.fiber_cropped.copy()
            
            if self.apply_smooth.get():
                window_size = self.smooth_window.get()
                poly_order = self.smooth_order.get()
                
                if window_size % 2 == 0:
                    window_size += 1
                    
                for channel_num in self.active_channels:
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in self.preprocessed_data.columns:
                            smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            self.preprocessed_data[smoothed_col] = savgol_filter(
                                self.preprocessed_data[target_col], window_size, poly_order)
            
            if self.apply_baseline.get():
                start_time = self.baseline_start.get()
                end_time = self.baseline_end.get()
                self.baseline_period = [start_time, end_time]
                
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                baseline_mask = (time_data >= start_time) & (time_data <= end_time)
                
                if not any(baseline_mask):
                    messagebox.showerror("Error", "No data in the specified baseline period")
                    return
                    
                for channel_num in self.active_channels:
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if not target_col or target_col not in self.preprocessed_data.columns:
                            continue
                        
                        smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                        if smoothed_col in self.preprocessed_data.columns:
                            signal_col = smoothed_col
                        else:
                            signal_col = target_col
                        
                        baseline_data = self.preprocessed_data.loc[baseline_mask, signal_col]
                        
                        if self.baseline_model.get() == "Exponential":
                            def exp_model(x, a, b, c):
                                return a * np.exp(-b * x) + c
                            
                            x0 = time_data[baseline_mask].values - time_data[baseline_mask].min()
                            y0 = baseline_data.values
                            
                            try:
                                params, _ = curve_fit(exp_model, x0, y0, p0=[max(y0), 0.1, min(y0)])
                                baseline_pred = exp_model(time_data.values - time_data.min(), *params)
                            except:
                                X = time_data[baseline_mask].values.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(X, baseline_data.values)
                                baseline_pred = model.predict(time_data.values.reshape(-1, 1))
                        else:
                            X = time_data[baseline_mask].values.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, baseline_data.values)
                            baseline_pred = model.predict(time_data.values.reshape(-1, 1))
                        
                        baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                        self.preprocessed_data[baseline_corrected_col] = (
                            self.preprocessed_data[signal_col] - baseline_pred)
            
            if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                self.motion_correction()
            
            messagebox.showinfo("Success", "Preprocessing applied successfully")
            self.set_status("Preprocessing applied")
            
            self.plot_preprocessed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.set_status("Preprocessing failed")
    
    def show_step3(self):
        if self.preprocessed_data is None:
            messagebox.showwarning("No Data", "Please preprocess data first")
            return
            
        self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(self.step3_frame, text="Calculate and plot ΔF/F", command=self.calculate_and_plot_dff).pack(fill="x", padx=5, pady=2)
        ttk.Button(self.step3_frame, text="Calculate and plot Z-score", command=self.calculate_and_plot_zscore).pack(fill="x", padx=5, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")
        # self.clear_step_frames()
        
        self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        param_frame = ttk.Frame(self.step4_frame)
        param_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Pre-event Window (s):").grid(row=0, column=0, sticky="w")
        self.pre_event = ttk.Entry(param_frame, width=8)
        self.pre_event.insert(0, "10")
        self.pre_event.grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="Post-event Window (s):").grid(row=1, column=0, sticky="w")
        self.post_event = ttk.Entry(param_frame, width=8)
        self.post_event.insert(0, "20")
        self.post_event.grid(row=1, column=1, padx=5)
        
        ttk.Label(param_frame, text="Event Type:").grid(row=2, column=0, sticky="w")
        self.event_type = ttk.Entry(param_frame, width=8)
        self.event_type.grid(row=2, column=1, padx=5)
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Plot Event-Related Activity", 
                  command=self.plot_event_related_activity).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Plot Activity Heatmap", 
                  command=self.plot_heatmap).pack(side=tk.LEFT, padx=5)
    
    def plot_preprocessed(self):
        """Plot preprocessed data"""
        if self.preprocessed_data is None:
            messagebox.showwarning("No Data", "Please preprocess data first")
            return
            
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if target_col and target_col in self.preprocessed_data.columns:
                        ax.plot(time_data, self.preprocessed_data[target_col], 
                               'b-', alpha=0.3, label=f'CH{channel_num} Raw Signal')
                
                smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                if smoothed_col in self.preprocessed_data.columns:
                    ax.plot(time_data, self.preprocessed_data[smoothed_col], 
                           'g-', alpha=0.7, label=f'CH{channel_num} Smoothed')
                
                baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                if baseline_corrected_col in self.preprocessed_data.columns:
                    ax.plot(time_data, self.preprocessed_data[baseline_corrected_col], 
                           'r-', label=f'CH{channel_num} Baseline Corrected')
                
                motion_corrected_col = f"CH{channel_num}_motion_corrected"
                if motion_corrected_col in self.preprocessed_data.columns:
                    ax.plot(time_data, self.preprocessed_data[motion_corrected_col], 
                           'm-', label=f'CH{channel_num} Motion Corrected')
            
            ax.set_title("Preprocessed Fiber Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.legend()
            self.canvas.draw()
            self.set_status("Preprocessed data plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot preprocessed data: {str(e)}")
            self.set_status("Preprocessed plot failed")
    
    def load_eztrack_Results(self):
        """Load ezTrack (FreezeAnalysis) results"""
        self.eztrack_results_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.eztrack_results_path:
            filename = os.path.basename(self.eztrack_results_path)
            messagebox.showinfo("ezTrack Loaded", f"Selected ezTrack file:\n{filename}")
            self.set_status(f"ezTrack results loaded: {filename}")

    def load_events(self):
        """Load event CSV file (sound and shock timings)"""
        super().load_events()

    def analyze(self):
        if not all([self.eztrack_results_path, self.events_path]):
            messagebox.showwarning("Missing Inputs", "Please load ezTrack results and events file first.")
            return
        
        if self.event_time_absolute:
            if self.include_fiber and self.video_start_fiber is not None:
                self.event_data['start_time'] = self.event_data['start_absolute'] - self.video_start_fiber
                self.event_data['end_time'] = self.event_data['end_absolute'] - self.video_start_fiber
                self.event_time_absolute = False
            else:
                min_start = self.event_data['start_absolute'].min()
                self.event_data['start_time'] = self.event_data['start_absolute'] - min_start
                self.event_data['end_time'] = self.event_data['end_absolute'] - min_start
                self.event_time_absolute = False
        
        self.set_status("Running freezing analysis...")
        try:
            self.eztrack_result = pd.read_csv(self.eztrack_results_path)
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
            event_start = event['start_time'] * fps
            event_end = event['end_time'] * fps
            
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
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            for _, row in self.event_data.iterrows():
                color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                ax.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)
            
            x = self.window_freeze['time'].values
            y = self.window_freeze['freeze_sec'].values
            
            ax.plot(x, y, marker='o', color='black', label='Freezing')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Freezing Duration (s) per 30s window")
            ax.set_title("Freezing Timeline with Events")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
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
            event_type = event['Event Type']
            if event_type == 0:
                event_type_str = "Wait"
            elif event_type == 1:
                event_type_str = "Sound"
            elif event_type == 2:
                event_type_str = "Shock"

            results.append({
                'Event': f"Event {idx+1}",
                'Start Time': event['start_time'],
                'End Time': event['end_time'],
                'Event Type': event_type_str,
                'Freezing Percentage': self.freezing_data.get(event_name, 0)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Exported", f"Freezing results saved to:\n{save_path}")
        self.set_status("Freezing results exported.")
    
    def plot_freezing_with_fiber(self):
        """Plot freezing data with fiber photometry data (cropped to video markers)"""
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run freezing analysis first")
            return
        if self.fiber_cropped is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        if self.video_start_fiber is None or self.video_end_fiber is None:
            messagebox.showwarning("Missing Markers", "Please set video start/end markers first")
            return
        
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Convert freezing data to absolute time
            freezing_abs_time = self.video_start_fiber + self.window_freeze['time']
            
            # Plot freezing data (in absolute time)
            ax.plot(freezing_abs_time, self.window_freeze['freeze_sec'], 
                    marker='o', color='black', label='Freezing (s)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Freezing Duration (s) per 30s window', color='black')
            
            # Plot fiber data on second axis (cropped to video markers)
            time_col = self.channels['time']
            time_data = self.fiber_cropped[time_col]
            
            ax2 = ax.twinx()
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if target_col and target_col in self.fiber_cropped.columns:
                        color = colors[i % len(colors)]
                        ax2.plot(time_data, self.fiber_cropped[target_col], 
                               f'{color}-', linewidth=1, label=f'CH{channel_num} Fiber Data')
                        ax2.set_ylabel('Fiber Signal', color=f'{color}')
            
            # Add events if available (convert to absolute time)
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    
                    if self.event_time_absolute:
                        start_abs = row[start_col]
                        end_abs = row[end_col]
                    else:
                        start_abs = self.video_start_fiber + row[start_col]
                        end_abs = self.video_start_fiber + row[end_col]
                    
                    ax.axvspan(start_abs, end_abs, color=color, alpha=0.3)
            
            ax.set_title('Freezing Analysis with Fiber Photometry')
            ax.grid(False)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            self.canvas.draw()
            self.set_status("Freezing with fiber plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot freezing with fiber: {str(e)}")
            self.set_status("Plot failed")

class PupilAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("Pupil Analyzer")
        self.dlc_results_path = ""
        self.timestamp_path = ""
        self.ast2_path = ""
        self.pupil_data = {}
        self.aligned_data = {}
        self.cam_id = None
        self.experiment_start = None
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(control_panel, text="One-click Import", command=self.one_click_import, 
                  style="Accent.TButton").pack(fill="x", padx=5, pady=10)
        
        pupil_frame = ttk.LabelFrame(control_panel, text="Pupil Analysis")
        pupil_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(pupil_frame, text="Load DLC Results", command=self.load_dlc_Results).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Load Timestamp CSV", command=self.load_timestamps).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Load Events CSV", command=self.load_events).grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Load AST2 File", command=self.load_ast2_file).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Run Pupil Analysis", command=self.run_pupil_analysis).grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Align AST2 Data", command=self.align_ast2_data).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Plot Pupil Distance", command=self.plot_pupil_distance).grid(row=3, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Plot Combined Data", command=self.plot_combined_data).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(fiber_frame, text="Load Fiber Data", command=self.load_fiber_data).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Align Fiber Data", command=self.set_video_markers).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=1, column=0, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Plot Pupil w/Fiber", command=self.plot_pupil_with_fiber).grid(row=1, column=1, sticky="ew", padx=5, pady=2)

    def show_step1(self):
        self.set_status("Step 1: Data Loading")
    
    def show_step2(self):
        self.set_status("Step 2: Preprocessing")
        # self.clear_step_frames()
        
        self.step2_frame = ttk.LabelFrame(self.control_frame, text="Preprocessing")
        self.step2_frame.pack(fill="x", padx=5, pady=5)
        
        signal_frame = ttk.Frame(self.step2_frame)
        signal_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(signal_frame, text="Target Signal:").grid(row=0, column=0, sticky="w")
        target_options = ["470", "560"]
        target_menu = ttk.OptionMenu(signal_frame, self.target_signal_var, "470", *target_options)
        target_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        ttk.Label(signal_frame, text="Reference Signal:").grid(row=1, column=0, sticky="w")
        ref_options = ["410", "baseline"]
        ref_menu = ttk.OptionMenu(signal_frame, self.reference_signal_var, "410", *ref_options)
        ref_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        self.baseline_frame = ttk.Frame(self.step2_frame)
        self.baseline_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(self.baseline_frame, text="Baseline Period (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_start, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(self.baseline_frame, text="to").grid(row=0, column=2)
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_end, width=6).grid(row=0, column=3, padx=2)
        
        if self.reference_signal_var.get() != "baseline":
            self.baseline_frame.pack_forget()
        
        model_frame = ttk.Frame(self.step2_frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_frame, text="Baseline Model:").grid(row=0, column=0, sticky="w")
        model_options = ["Polynomial", "Exponential"]
        model_menu = ttk.OptionMenu(model_frame, self.baseline_model, "Polynomial", *model_options)
        model_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        smooth_frame = ttk.Frame(self.step2_frame)
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(smooth_frame, text="Apply Smoothing", variable=self.apply_smooth,
                       command=lambda: self.toggle_widgets(smooth_frame, self.apply_smooth.get(), 1)).grid(row=0, column=0, sticky="w")
        
        param_frame = ttk.Frame(smooth_frame)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Label(param_frame, text="Window Size:").grid(row=0, column=0, sticky="w")
        ttk.Scale(param_frame, from_=3, to=101, orient=tk.HORIZONTAL, 
                 length=150, variable=self.smooth_window,
                 command=lambda v: self.smooth_window.set(int(float(v)))).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_window).grid(row=0, column=2)
        
        ttk.Label(param_frame, text="Polynomial Order:").grid(row=1, column=0, sticky="w")
        ttk.Scale(param_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                 length=150, variable=self.smooth_order,
                 command=lambda v: self.smooth_order.set(int(float(v)))).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_order).grid(row=1, column=2)
        
        param_frame.grid_remove()
        
        baseline_corr_frame = ttk.Frame(self.step2_frame)
        baseline_corr_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=self.apply_baseline).grid(row=0, column=0, sticky="w")
        
        motion_frame = ttk.Frame(self.step2_frame)
        motion_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(motion_frame, text="Apply Motion Correction", variable=self.apply_motion,
                       command=lambda: self.toggle_motion_correction()).grid(row=0, column=0, sticky="w")
        
        ttk.Button(self.step2_frame, text="Apply Preprocessing and Plot", 
                  command=self.apply_preprocessing).pack(pady=10)
        
        self.reference_signal_var.trace_add("write", self.update_baseline_ui)
    
    def update_baseline_ui(self, *args):
        if self.reference_signal_var.get() == "baseline":
            self.baseline_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.baseline_frame.pack_forget()
    
    def toggle_motion_correction(self):
        if self.reference_signal_var.get() != "410":
            self.apply_motion.set(False)
            messagebox.showinfo("Info", "Motion correction requires 410nm as reference signal")
    
    def toggle_widgets(self, parent_frame, show, index):
        children = parent_frame.winfo_children()
        if len(children) > index:
            if show:
                children[index].grid()
            else:
                children[index].grid_remove()
    
    def apply_preprocessing(self):
        if self.fiber_cropped is None:
            messagebox.showwarning("No Data", "Please load fiber data first")
            return
            
        try:
            self.preprocessed_data = self.fiber_cropped.copy()
            
            if self.apply_smooth.get():
                window_size = self.smooth_window.get()
                poly_order = self.smooth_order.get()
                
                if window_size % 2 == 0:
                    window_size += 1
                    
                for channel_num in self.active_channels:
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in self.preprocessed_data.columns:
                            smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            self.preprocessed_data[smoothed_col] = savgol_filter(
                                self.preprocessed_data[target_col], window_size, poly_order)
            
            if self.apply_baseline.get():
                start_time = self.baseline_start.get()
                end_time = self.baseline_end.get()
                self.baseline_period = [start_time, end_time]
                
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                baseline_mask = (time_data >= start_time) & (time_data <= end_time)
                
                if not any(baseline_mask):
                    messagebox.showerror("Error", "No data in the specified baseline period")
                    return
                    
                for channel_num in self.active_channels:
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if not target_col or target_col not in self.preprocessed_data.columns:
                            continue
                        
                        smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                        if smoothed_col in self.preprocessed_data.columns:
                            signal_col = smoothed_col
                        else:
                            signal_col = target_col
                        
                        baseline_data = self.preprocessed_data.loc[baseline_mask, signal_col]
                        
                        if self.baseline_model.get() == "Exponential":
                            def exp_model(x, a, b, c):
                                return a * np.exp(-b * x) + c
                            
                            x0 = time_data[baseline_mask].values - time_data[baseline_mask].min()
                            y0 = baseline_data.values
                            
                            try:
                                params, _ = curve_fit(exp_model, x0, y0, p0=[max(y0), 0.1, min(y0)])
                                baseline_pred = exp_model(time_data.values - time_data.min(), *params)
                            except:
                                X = time_data[baseline_mask].values.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(X, baseline_data.values)
                                baseline_pred = model.predict(time_data.values.reshape(-1, 1))
                        else:
                            X = time_data[baseline_mask].values.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, baseline_data.values)
                            baseline_pred = model.predict(time_data.values.reshape(-1, 1))
                        
                        baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                        self.preprocessed_data[baseline_corrected_col] = (
                            self.preprocessed_data[signal_col] - baseline_pred)
            
            if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                self.motion_correction()
            
            messagebox.showinfo("Success", "Preprocessing applied successfully")
            self.set_status("Preprocessing applied")
            
            self.plot_preprocessed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.set_status("Preprocessing failed")
    
    def show_step3(self):
        if self.preprocessed_data is None:
            messagebox.showwarning("No Data", "Please preprocess data first")
            return
            
        self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(self.step3_frame, text="Calculate and Plot ΔF/F", command=self.calculate_and_plot_dff).pack(fill="x", padx=5, pady=2)
        ttk.Button(self.step3_frame, text="Calculate and Plot Z-score", command=self.calculate_and_plot_zscore).pack(fill="x", padx=5, pady=2)
        ttk.Button(self.step3_frame, text="Plot ΔF/F", command=self.plot_dff).pack(fill="x", padx=5, pady=2)
        ttk.Button(self.step3_frame, text="Plot Z-score", command=self.plot_zscore).pack(fill="x", padx=5, pady=2)
        
        self.set_status("Step 3: ΔF/F & Z-score Calculation")
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")
        # self.clear_step_frames()
        
        self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        param_frame = ttk.Frame(self.step4_frame)
        param_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Pre-event Window (s):").grid(row=0, column=0, sticky="w")
        self.pre_event = ttk.Entry(param_frame, width=8)
        self.pre_event.insert(0, "10")
        self.pre_event.grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="Post-event Window (s):").grid(row=1, column=0, sticky="w")
        self.post_event = ttk.Entry(param_frame, width=8)
        self.post_event.insert(0, "20")
        self.post_event.grid(row=1, column=1, padx=5)
        
        ttk.Label(param_frame, text="Event Type:").grid(row=2, column=0, sticky="w")
        self.event_type = ttk.Entry(param_frame, width=8)
        self.event_type.grid(row=2, column=1, padx=5)
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Plot Event-Related Activity", 
                  command=self.plot_event_related_activity).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Plot Activity Heatmap", 
                  command=self.plot_heatmap).pack(side=tk.LEFT, padx=5)
    
    def plot_preprocessed(self):
        """Plot preprocessed data"""
        if self.preprocessed_data is None:
            messagebox.showwarning("No Data", "Please preprocess data first")
            return
            
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if target_col and target_col in self.preprocessed_data.columns:
                        ax.plot(time_data, self.preprocessed_data[target_col], 
                               'b-', alpha=0.3, label=f'CH{channel_num} Raw Signal')
                
                # 绘制平滑数据
                smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                if smoothed_col in self.preprocessed_data.columns:
                    ax.plot(time_data, self.preprocessed_data[smoothed_col], 
                           'g-', alpha=0.7, label=f'CH{channel_num} Smoothed')
                
                baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                if baseline_corrected_col in self.preprocessed_data.columns:
                    ax.plot(time_data, self.preprocessed_data[baseline_corrected_col], 
                           'r-', label=f'CH{channel_num} Baseline Corrected')
                
                motion_corrected_col = f"CH{channel_num}_motion_corrected"
                if motion_corrected_col in self.preprocessed_data.columns:
                    ax.plot(time_data, self.preprocessed_data[motion_corrected_col], 
                           'm-', label=f'CH{channel_num} Motion Corrected')
            
            ax.set_title("Preprocessed Fiber Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.legend()
            self.canvas.draw()
            self.set_status("Preprocessed data plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot preprocessed data: {str(e)}")
            self.set_status("Preprocessed plot failed")
    
    def plot_dff(self):
        """Plot ΔF/F data"""
        if self.dff_data is None:
            messagebox.showwarning("No Data", "Please calculate ΔF/F first")
            return
            
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.dff_data:
                    dff_col = f"CH{channel_num}_dff"
                    if dff_col in self.preprocessed_data.columns:
                        color = colors[i % len(colors)]
                        ax.plot(time_data, self.preprocessed_data[dff_col], 
                               f'{color}-', label=f'CH{channel_num} ΔF/F')
            
            ax.axvspan(self.baseline_period[0], self.baseline_period[1], color='green', alpha=0.2)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.set_title("ΔF/F")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔF/F")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
            self.set_status("ΔF/F plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot ΔF/F: {str(e)}")
            self.set_status("ΔF/F plot failed")
    
    def plot_zscore(self):
        """Plot Z-score data"""
        if self.zscore_data is None:
            messagebox.showwarning("No Data", "Please calculate Z-score first")
            return
            
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            time_col = self.channels['time']
            time_data = self.preprocessed_data[time_col] - self.video_start_fiber
            
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.dff_data:
                    zscore_col = f"CH{channel_num}_zscore"
                    if zscore_col in self.preprocessed_data.columns:
                        color = colors[i % len(colors)]
                        ax.plot(time_data, self.preprocessed_data[zscore_col], 
                               f'{color}-', label=f'CH{channel_num} Z-score')
            
            ax.axvspan(self.baseline_period[0], self.baseline_period[1], color='green', alpha=0.2)
            
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - self.video_start_fiber
                    end_time = row[end_col] - self.video_start_fiber
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.set_title("Z-score")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Z-score")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
            self.set_status("Z-score plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot Z-score: {str(e)}")
            self.set_status("Z-score plot failed")
    
    def load_dlc_Results(self):
        """Load DLC results and extract camera ID from filename"""
        self.dlc_results_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.dlc_results_path:
            filename = os.path.basename(self.dlc_results_path)
            match = re.search(r'cam(\d+)', filename)
            if match:
                self.cam_id = int(match.group(1))
                messagebox.showinfo("DLC Loaded", f"Selected DLC for Camera {self.cam_id}\n{filename}")
                self.set_status(f"DLC results loaded for Camera {self.cam_id}")
            else:
                messagebox.showerror("Error", "Could not find camera ID in filename (expected 'camX')")
                self.dlc_results_path = ""

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

    def load_ast2_file(self):
        self.ast2_path = filedialog.askopenfilename(filetypes=[("AST2 files", "*.ast2")])
        if self.ast2_path:
            self.set_status("Reading AST2 file...")
            try:
                self.ast2_data = self.h_AST2_readData(self.ast2_path)
                messagebox.showinfo("AST2 Loaded", "AST2 file successfully loaded and parsed!")
                self.set_status("AST2 file loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load AST2 file: {str(e)}")
                self.set_status("AST2 load failed.")

    def h_AST2_readData(self, filename):
        """Read AST2 file format and return header and data with absolute timestamps"""
        header = {}
        data = None
        
        with open(filename, 'rb') as fid:
            # Read header
            identifier = fid.read(4)
            if identifier != b'AST2':
                raise ValueError("Not a valid AST2 file (invalid identifier)")
            
            version = struct.unpack('<H', fid.read(2))[0]
            header['version'] = version / 10.0
            
            header['num_channels'] = struct.unpack('<H', fid.read(2))[0]
            header['inputRate'] = struct.unpack('<f', fid.read(4))[0]
            header['speed_down_sample_factor'] = struct.unpack('<I', fid.read(4))[0]
            
            start_time_bytes = fid.read(20)
            header['startTime'] = start_time_bytes.decode('utf-8').strip('\x00')
            header['reserved'] = fid.read(40)
            
            # Calculate data size
            fid.seek(0, 2)
            file_size = fid.tell()
            header_size = 4 + 2 + 2 + 4 + 4 + 20 + 40
            data_size = file_size - header_size
            
            # Read data
            fid.seek(header_size)
            num_samples = data_size // (8 + 4 * header['num_channels'])
            if num_samples <= 0:
                raise ValueError("No data found in AST2 file")
            
            timestamps = []
            speed_data = [[] for _ in range(header['num_channels'])]
            
            for _ in range(num_samples):
                timestamp = struct.unpack('<d', fid.read(8))[0]
                timestamps.append(timestamp)
                
                for ch in range(header['num_channels']):
                    speed = struct.unpack('<f', fid.read(4))[0]
                    speed_data[ch].append(speed)
            if invert == True:
                data = {
                    'timestamps': np.array(timestamps),
                    'speed': -np.array(speed_data)
                }
            else:
                data = {
                    'timestamps': np.array(timestamps),
                    'speed': np.array(speed_data)
                }
        
        return {'header': header, 'data': data}

    def align_ast2_data(self):
        """Align AST2 data using camera ID from DLC filename"""
        if not all([self.ast2_path, self.ast2_data, self.timestamp_path, self.cam_id is not None]):
            messagebox.showwarning("Missing Data", "Please load AST2 file, timestamp file and DLC results first")
            return
        
        try:
            self.set_status("Aligning AST2 data...")
            timestamps = pd.read_csv(self.timestamp_path)
            
            # Get camera and speed sensor timestamps
            cam_device = f"Camera {self.cam_id}"
            speed_device = "Speed Sensor"
            
            # Get start times for alignment
            cam_start = timestamps[(timestamps['Device'] == cam_device) & 
                                 (timestamps['Action'] == 'Start')]['Timestamp'].values[0]
            speed_start = timestamps[(timestamps['Device'] == speed_device) & 
                                   (timestamps['Action'] == 'Start')]['Timestamp'].values[0]
            
            # Calculate time offset between speed sensor and camera
            time_offset = cam_start - speed_start
            
            # Convert AST2 timestamps to absolute time (matching camera time)
            ast2_absolute_time = self.ast2_data['data']['timestamps'] + speed_start + time_offset
            
            # Get valid data range based on experiment duration
            exp_start = self.experiment_start
            exp_end = timestamps[(timestamps['Device'] == 'Experiment') & 
                               (timestamps['Action'] == 'End')]['Timestamp'].values[0]
            valid_idx = (ast2_absolute_time >= exp_start) & (ast2_absolute_time <= exp_end)
            
            # Store aligned data
            self.aligned_data = {
                'time': ast2_absolute_time[valid_idx],
                'speed': self.ast2_data['data']['speed'][0][valid_idx],  # Use first channel
                'start_time': exp_start,
                'end_time': exp_end
            }
            
            messagebox.showinfo("Success", f"AST2 data aligned with Camera {self.cam_id} timeline!")
            self.set_status(f"AST2 data aligned with Camera {self.cam_id}")
            
        except Exception as e:
            messagebox.showerror("Alignment Error", f"Failed to align data: {str(e)}")
            self.set_status("Alignment failed.")

    def run_pupil_analysis(self):
        """Run pupil analysis with absolute time conversion"""
        if not all([self.dlc_results_path, self.timestamp_path, self.cam_id is not None]):
            messagebox.showwarning("Missing Inputs", "Please load DLC results and timestamps first")
            return
        
        self.set_status("Running pupil analysis...")
        try:
            # Read DLC results
            rawdata = pd.read_csv(self.dlc_results_path, low_memory=False)
            
            # Extract body parts and coordinates
            bodyparts = rawdata.iloc[0, 1::3].values
            coordinates = {}
            for i in range(len(bodyparts)):
                tmp = {}
                tmp['x'] = rawdata.iloc[2:, 3*i+1].values.astype(float)
                tmp['y'] = -rawdata.iloc[2:, 3*i+2].values.astype(float)  # Flip y-axis
                tmp['reliability'] = rawdata.iloc[2:, 3*i+3].values.astype(float)
                coordinates[bodyparts[i]] = tmp
            
            # Apply reliability threshold
            reliability_thresh = 0.99
            rlb_idx = {}
            for part in ['pupil_top', 'pupil_bottom']:
                rlb_idx[part] = np.where(coordinates[part]['reliability'] >= reliability_thresh)[0]
            
            # Use common reliable indices
            common_idx = np.intersect1d(rlb_idx['pupil_top'], rlb_idx['pupil_bottom'])
            
            # Get camera start time for absolute time conversion
            timestamps = pd.read_csv(self.timestamp_path)
            cam_device = f"Camera {self.cam_id}"
            cam_start = timestamps[(timestamps['Device'] == cam_device) & 
                                 (timestamps['Action'] == 'Start')]['Timestamp'].values[0]
            
            # Calculate time vectors (convert to absolute time)
            fps = 90
            relative_time = np.arange(0, len(coordinates['pupil_top']['x'])) / fps
            absolute_time = cam_start + relative_time  # Convert to absolute time
            time_span_r = absolute_time[common_idx]
            
            # Extract reliable coordinates
            pupil_top_x_r = coordinates['pupil_top']['x'][common_idx]
            pupil_top_y_r = coordinates['pupil_top']['y'][common_idx]
            pupil_bottom_x_r = coordinates['pupil_bottom']['x'][common_idx]
            pupil_bottom_y_r = coordinates['pupil_bottom']['y'][common_idx]
            
            # Calculate pupil distance
            pupil_dist = np.sqrt((pupil_top_x_r - pupil_bottom_x_r)**2 + 
                                (pupil_top_y_r - pupil_bottom_y_r)** 2)
            
            # Store results with absolute time
            self.pupil_data = {
                'time': time_span_r,
                'distance': pupil_dist,
                'coordinates': coordinates,
                'common_idx': common_idx
            }
            
            if not self.event_time_absolute:
                self.event_data['start_absolute'] = cam_start + self.event_data['start_time']
                self.event_data['end_absolute'] = cam_start + self.event_data['end_time']
                self.event_time_absolute = True
                self.set_status("Converted event times to absolute using camera start")
            
            messagebox.showinfo("Success", "Pupil analysis completed!")
            self.set_status("Pupil analysis completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Pupil analysis failed: {str(e)}")
            self.set_status("Pupil analysis failed.")

    def plot_pupil_distance(self):
        if not self.pupil_data:
            messagebox.showwarning("No Data", "Please run pupil analysis first.")
            return
        
        if not self.events_path:
            messagebox.showwarning("Missing Events", "Please load events CSV first.")
            return
        
        try:
            self.set_status("Plotting pupil distance...")
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            if self.event_time_absolute:
                start_col = 'start_absolute'
                end_col = 'end_absolute'
            else:
                start_col = 'start_time'
                end_col = 'end_time'
            
            for _, row in self.event_data.iterrows():
                color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                ax.axvspan(row[start_col], row[end_col], color=color, alpha=0.3)
            
            # Plot pupil distance
            ax.plot(self.pupil_data['time'], self.pupil_data['distance'], 
                    linewidth=1.5, label='Pupil Distance')
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pupil Distance (pixels)")
            ax.set_title("Pupil Distance Timeline with Events")
            ax.grid(False)
            ax.legend()
            
            self.canvas.draw()
            self.set_status("Pupil plot completed.")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot pupil distance: {str(e)}")
            self.set_status("Plot failed.")

    def plot_combined_data(self):
        """Plot combined graph with pupil distance, speed, and event timelines"""
        if not all([self.aligned_data, self.pupil_data, self.event_data is not None]):
            messagebox.showwarning("No Data", "Please align AST2 data, run pupil analysis and load events first")
            return

        try:
            self.set_status("Plotting combined data...")
            
            first_event_start = self.event_data['start_absolute'].min()
            event_duration = self.event_data['end_absolute'].max() - first_event_start

            def to_relative_time(absolute_time):
                return absolute_time - first_event_start

            ast2_relative_time = to_relative_time(self.aligned_data['time'])
            ast2_valid_idx = (ast2_relative_time >= 0) & (ast2_relative_time <= event_duration)
            cropped_ast2_time = ast2_relative_time[ast2_valid_idx]
            cropped_ast2_speed = self.aligned_data['speed'][ast2_valid_idx]

            pupil_relative_time = to_relative_time(self.pupil_data['time'])
            pupil_valid_idx = (pupil_relative_time >= 0) & (pupil_relative_time <= event_duration)
            cropped_pupil_time = pupil_relative_time[pupil_valid_idx]
            cropped_pupil_distance = self.pupil_data['distance'][pupil_valid_idx]

            self.fig.clear()
            ax1 = self.fig.add_subplot(111)
            ax2 = ax1.twinx()

            for _, row in self.event_data.iterrows():
                start_rel = to_relative_time(row['start_absolute'])
                end_rel = to_relative_time(row['end_absolute'])
                color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                ax1.axvspan(start_rel, end_rel, color=color, alpha=0.3)

            ax1.plot(cropped_ast2_time, cropped_ast2_speed, 
                    linewidth=1.5, color='blue', label='Speed (degrees/s)')
            ax1.set_xlabel("Time Relative to First Event (s)")
            ax1.set_ylabel("Speed (degrees/s)", color='blue')

            ax2.plot(cropped_pupil_time, cropped_pupil_distance, 
                    linewidth=1.5, color='red', label='Pupil Distance (pixels)')
            ax2.set_ylabel("Pupil Distance (pixels)", color='red')
            
            # Add fiber data if available
            if self.fiber_data is not None and self.video_start_fiber is not None and self.video_end_fiber is not None:
                time_col = self.channels['time']
                fiber_in_range = self.fiber_data[
                    (self.fiber_data[time_col] >= self.video_start_fiber) & 
                    (self.fiber_data[time_col] <= self.video_end_fiber)]
                
                if not fiber_in_range.empty:
                    fiber_rel_time = to_relative_time(fiber_in_range[time_col])
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.plot(fiber_rel_time, fiber_in_range[self.channel_data[1]['470']], 
                            linewidth=1.5, color='purple', label='Fiber Signal')
                    ax3.set_ylabel("Fiber Signal", color='purple')
            
            ax1.set_title("Pupil Distance, Speed and Fiber Signal Timeline")
            ax1.grid(False)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            self.canvas.draw()
            self.set_status("Combined plot completed.")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot combined data: {str(e)}")
            self.set_status("Plot failed.")
    
    def plot_pupil_with_fiber(self):
        """Plot pupil data with fiber photometry data"""
        if not self.pupil_data:
            messagebox.showwarning("No Data", "Please run pupil analysis first")
            return
        if self.fiber_cropped is None or not self.active_channels:
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            self.fig.clear()
            ax1 = self.fig.add_subplot(111)
            
            # Plot pupil data
            ax1.plot(self.pupil_data['time'], self.pupil_data['distance'], 
                    color='red', label='Pupil Distance')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pupil Distance (pixels)', color='red')
            
            # Plot fiber data on second axis
            time_col = self.channels['time']
            time_data = self.fiber_cropped[time_col] - self.video_start_fiber
            ax2 = ax1.twinx()
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, channel_num in enumerate(self.active_channels):
                if channel_num in self.channel_data:
                    target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                    if target_col and target_col in self.fiber_cropped.columns:
                        color = colors[i % len(colors)]
                        ax2.plot(time_data, self.fiber_cropped[target_col], 
                               f'{color}-', linewidth=1, label=f'CH{channel_num} Fiber Data')
                        ax2.set_ylabel('Fiber Signal', color=f'{color}')
            
            # Add events if available
            if self.event_data is not None:
                if self.event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    ax1.axvspan(row[start_col], row[end_col], color=color, alpha=0.3)
            
            ax1.set_title('Pupil Analysis with Fiber Photometry')
            ax1.grid(False)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            self.canvas.draw()
            self.set_status("Pupil with fiber plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot pupil with fiber: {str(e)}")
            self.set_status("Plot failed")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModeSelectionApp(root)
    root.mainloop()