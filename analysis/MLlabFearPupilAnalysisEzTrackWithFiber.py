import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import struct
import re
import os
import math
from datetime import datetime
from itertools import groupby
import matplotlib.dates as mdates

invert = True

class FreezingAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Conditioning Analyzer")
        self.root.geometry("200x700")  # Adjusted height for better layout
        self.root.resizable(False, False)
        
        # Existing data paths
        self.eztrack_results_path = ""
        self.dlc_results_path = ""
        self.timestamp_path = ""
        self.ast2_path = ""
        self.events_path = ""
        
        # Fiber photometry data
        self.fiber_data_path = ""
        self.fiber_data = None
        self.video_start_fiber = None
        self.video_end_fiber = None
        
        # Data storage
        self.freezing_data = {}
        self.pupil_data = {}
        self.ast2_data = {}
        self.aligned_data = {}
        self.cam_id = None
        self.experiment_start = None
        self.event_data = None
        self.current_mode = None
        self.window_freeze = None
        
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky="NSEW")
        
        self.create_button(main_frame, "Freezing Analysis", self.select_freezing_mode, 0)
        self.create_button(main_frame, "Pupil Analysis", self.select_pupil_mode, 1)
        
        # Freezing frame
        self.freezing_frame = ttk.LabelFrame(main_frame, text="Freezing Analysis")
        self.freezing_frame.grid(row=2, column=0, pady=10, sticky="EW")
        
        self.create_button(self.freezing_frame, "Load ezTrack Results", self.load_eztrack_Results, 0)
        self.create_button(self.freezing_frame, "Load Events CSV", self.load_events, 1)
        self.create_button(self.freezing_frame, "Run Freezing Analysis", self.analyze, 2)
        self.create_button(self.freezing_frame, "Plot Freezing Timeline", self.plot_freezing_windows, 3)
        self.create_button(self.freezing_frame, "Export Results to CSV", self.export_freezing_csv, 4)
        
        # Fiber options for freezing analysis
        self.freezing_fiber_frame = ttk.LabelFrame(self.freezing_frame, text="Fiber Photometry Options")
        self.freezing_fiber_frame.grid(row=5, column=0, pady=10, sticky="EW")
        
        self.create_button(self.freezing_fiber_frame, "Load Fiber Data", self.load_fiber_data, 0)
        self.create_button(self.freezing_fiber_frame, "Set Video Start/End", self.set_video_markers, 1)
        self.create_button(self.freezing_fiber_frame, "Plot Fiber Data", self.plot_fiber_data, 2)
        self.create_button(self.freezing_fiber_frame, "Plot Freezing w/Fiber", self.plot_freezing_with_fiber, 3)
        
        # Pupil frame
        self.pupil_frame = ttk.LabelFrame(main_frame, text="Pupil Analysis")
        self.pupil_frame.grid(row=3, column=0, pady=10, sticky="EW")
        
        self.create_button(self.pupil_frame, "Load DLC Results", self.load_dlc_Results, 0)
        self.create_button(self.pupil_frame, "Load Timestamp CSV", self.load_timestamps, 1)
        self.create_button(self.pupil_frame, "Load Events CSV", self.load_events, 2)
        self.create_button(self.pupil_frame, "Load AST2 File", self.load_ast2_file, 3)
        self.create_button(self.pupil_frame, "Run Pupil Analysis", self.run_pupil_analysis, 4)
        self.create_button(self.pupil_frame, "Align AST2 Data", self.align_ast2_data, 5)
        self.create_button(self.pupil_frame, "Plot Pupil Distance", self.plot_pupil_distance, 6)
        self.create_button(self.pupil_frame, "Plot Combined Data", self.plot_combined_data, 7)
        
        # Fiber options for pupil analysis
        self.pupil_fiber_frame = ttk.LabelFrame(self.pupil_frame, text="Fiber Photometry Options")
        self.pupil_fiber_frame.grid(row=8, column=0, pady=10, sticky="EW")
        
        self.create_button(self.pupil_fiber_frame, "Load Fiber Data", self.load_fiber_data, 0)
        self.create_button(self.pupil_fiber_frame, "Set Video Start/End", self.set_video_markers, 1)
        self.create_button(self.pupil_fiber_frame, "Plot Fiber Data", self.plot_fiber_data, 2)
        self.create_button(self.pupil_fiber_frame, "Plot Pupil w/Fiber", self.plot_pupil_with_fiber, 3)
        
        # Hide all frames initially
        self.freezing_frame.grid_remove()
        self.pupil_frame.grid_remove()
        
        self.status_var = tk.StringVar()
        self.status_var.set("Select analysis mode")
        self.status_label = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="EW")

    def create_button(self, frame, text, command, row):
        btn = ttk.Button(frame, text=text, command=command)
        btn.grid(row=row, column=0, pady=6, sticky="EW")
        return btn

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def select_freezing_mode(self):
        self.current_mode = "freezing"
        self.freezing_frame.grid()
        self.pupil_frame.grid_remove()
        self.set_status("Freezing analysis mode selected")
    
    def select_pupil_mode(self):
        self.current_mode = "pupil"
        self.pupil_frame.grid()
        self.freezing_frame.grid_remove()
        self.set_status("Pupil analysis mode selected")

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
    
    def plot_fiber_data(self):
        """Plot fiber photometry data with dynamic channel handling"""
        if self.fiber_data is None:
            messagebox.showwarning("No Data", "Please load fiber data first")
            return
        
        try:
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
            
            fig, axs = plt.subplots(num_plots, 1, figsize=(12, 2 + 2*num_plots), sharex=True)
            if num_plots == 1:
                axs = [axs]  # Ensure axs is iterable
            
            plot_idx = 0
            
            # Plot available raw signals
            if green_col:
                axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data[green_col], 
                                'g-', label='470nm Signal')
                axs[plot_idx].set_ylabel('470nm')
                axs[plot_idx].legend(loc='upper right')
                plot_idx += 1
                
            if red_col:
                axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data[red_col], 
                                'r-', label='560nm Signal')
                axs[plot_idx].set_ylabel('560nm')
                axs[plot_idx].legend(loc='upper right')
                plot_idx += 1
                
            # Always plot isosbestic reference
            axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data[isos_col], 
                            'b-', label='410/415nm Reference')
            axs[plot_idx].set_ylabel('Reference')
            axs[plot_idx].legend(loc='upper right')
            plot_idx += 1
            
            # Plot normalized signals
            if 'normalized_signal_470' in self.fiber_data.columns and 'normalized_signal_560' in self.fiber_data.columns:
                axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_470'], 
                                'm-', label='470/410')
                axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_560'], 
                                'c-', label='560/410')
                axs[plot_idx].set_ylabel('Normalized')
                axs[plot_idx].legend(loc='upper right')
            elif 'normalized_signal_470' in self.fiber_data.columns:
                axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_470'], 
                                'm-', label='470/410')
                axs[plot_idx].set_ylabel('Normalized')
                axs[plot_idx].legend(loc='upper right')
            elif 'normalized_signal_560' in self.fiber_data.columns:
                axs[plot_idx].plot(self.fiber_data[time_col], self.fiber_data['normalized_signal_560'], 
                                'c-', label='560/410')
                axs[plot_idx].set_ylabel('Normalized')
                axs[plot_idx].legend(loc='upper right')
            
            axs[plot_idx].set_xlabel('Time (s)')
            
            # Highlight video period if markers are set
            if self.video_start_fiber is not None and self.video_end_fiber is not None:
                for ax in axs:
                    ax.axvspan(self.video_start_fiber, self.video_end_fiber, color='yellow', alpha=0.3)
                    ax.axvline(self.video_start_fiber, color='k', linestyle='--')
                    ax.axvline(self.video_end_fiber, color='k', linestyle='--')
            
            plt.suptitle(f'Fiber Photometry Data ({self.primary_signal} primary)')
            plt.tight_layout()
            plt.show()
            
            self.set_status("Fiber data plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot fiber data: {str(e)}")
            self.set_status("Fiber plot failed")
    
    def plot_freezing_with_fiber(self):
        """Plot freezing data with fiber photometry data (cropped to video markers)"""
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run freezing analysis first")
            return
        if self.fiber_data is None:
            messagebox.showwarning("No Fiber Data", "Please load fiber data first")
            return
        if self.video_start_fiber is None or self.video_end_fiber is None:
            messagebox.showwarning("Missing Markers", "Please set video start/end markers first")
            return
        
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Convert freezing data to absolute time
            freezing_abs_time = self.video_start_fiber + self.window_freeze['time']
            
            # Plot freezing data (in absolute time)
            ax1.plot(freezing_abs_time, self.window_freeze['freeze_sec'], 
                    marker='o', color='black', label='Freezing (s)')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Freezing Duration (s) per 30s window', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(False, linestyle='--', alpha=0.7)
            
            # Plot fiber data on second axis (cropped to video markers)
            time_col = self.channels['time']
            fiber_cropped = self.fiber_data[
                (self.fiber_data[time_col] >= self.video_start_fiber) & 
                (self.fiber_data[time_col] <= self.video_end_fiber)
            ]
            
            ax2 = ax1.twinx()
            ax2.plot(fiber_cropped[time_col], fiber_cropped['normalized_signal'], 
                    color='green', alpha=0.7, label='Fiber Signal')
            ax2.set_ylabel('Normalized Fiber Signal', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Add events if available (convert to absolute time)
            if self.event_data is not None:
                for _, row in self.event_data.iterrows():
                    start_abs = self.video_start_fiber + row['start_time']
                    end_abs = self.video_start_fiber + row['end_time']
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    ax1.axvspan(start_abs, end_abs, color=color, alpha=0.3)
            
            plt.title('Freezing Analysis with Fiber Photometry (Cropped to Video)')
            fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
            plt.tight_layout()
            plt.show()
            
            self.set_status("Freezing with fiber plotted (cropped to video)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot freezing with fiber: {str(e)}")
            self.set_status("Plot failed")
    
    def plot_pupil_with_fiber(self):
        """Plot pupil data with fiber photometry data"""
        if not self.pupil_data:
            messagebox.showwarning("No Data", "Please run pupil analysis first")
            return
        if self.fiber_data is None:
            messagebox.showwarning("No Fiber Data", "Please load fiber data first")
            return
        
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot pupil data
            ax1.plot(self.pupil_data['time'], self.pupil_data['distance'], 
                    color='red', label='Pupil Distance')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pupil Distance (pixels)', color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot fiber data on second axis
            time_col = next((col for col in self.fiber_data.columns if 'timestamp' in col.lower()), None)
            ax2 = ax1.twinx()
            ax2.plot(self.fiber_data[time_col], self.fiber_data['normalized_signal'], 
                    color='purple', alpha=0.7, label='Fiber Signal')
            ax2.set_ylabel('Normalized Fiber Signal', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            # Add events if available
            if self.event_data is not None:
                for _, row in self.event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    ax1.axvspan(row['start_absolute'], row['end_absolute'], color=color, alpha=0.3)
            
            plt.title('Pupil Analysis with Fiber Photometry')
            fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
            plt.tight_layout()
            plt.show()
            
            self.set_status("Pupil with fiber plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot pupil with fiber: {str(e)}")
            self.set_status("Plot failed")

    def load_eztrack_Results(self):
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
            
            plt.figure(figsize=(12, 5))
            
            for _, row in self.event_data.iterrows():
                color = "#b6b6b6ff" if row['Event Type'] == 0 else "#fcb500f0"
                plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)
            
            x = self.window_freeze['time'].values
            y = self.window_freeze['freeze_sec'].values
            
            plt.plot(x, y, marker='o', color='black', label='Freezing')
            plt.xlabel("Time (s)")
            plt.ylabel("Freezing Duration (s) per 30s window")
            plt.title("Freezing Timeline with Events")
            plt.grid(False)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
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
            
            plt.figure(figsize=(12, 6))
            
            # Add event backgrounds
            for _, row in self.event_data.iterrows():
                color = "#b6b6b6ff" if row['Event Type'] == 0 else "#fcb500f0"
                plt.axvspan(row['start_absolute'], row['end_absolute'], color=color, alpha=0.3)
            
            # Plot pupil distance
            plt.plot(self.pupil_data['time'], self.pupil_data['distance'], 
                    linewidth=1.5, label='Pupil Distance')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Pupil Distance (pixels)")
            plt.title("Pupil Distance Timeline with Events")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
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

            fig, ax1 = plt.subplots(figsize=(14, 7))
            ax2 = ax1.twinx()

            for _, row in self.event_data.iterrows():
                start_rel = to_relative_time(row['start_absolute'])
                end_rel = to_relative_time(row['end_absolute'])
                color = "#b6b6b6ff" if row['Event Type'] == 0 else "#fcb500f0"
                ax1.axvspan(start_rel, end_rel, color=color, alpha=0.3)

            ax1.plot(cropped_ast2_time, cropped_ast2_speed, 
                    linewidth=1.5, color='blue', label='Speed (degrees/s)')
            ax1.set_xlabel("Time Relative to First Event (s)")
            ax1.set_ylabel("Speed (degrees/s)", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2.plot(cropped_pupil_time, cropped_pupil_distance, 
                    linewidth=1.5, color='red', label='Pupil Distance (pixels)')
            ax2.set_ylabel("Pupil Distance (pixels)", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add fiber data if available
            if self.fiber_data is not None and self.video_start_fiber is not None and self.video_end_fiber is not None:
                time_col = next((col for col in self.fiber_data.columns if 'timestamp' in col.lower()), None)
                fiber_in_range = self.fiber_data[
                    (self.fiber_data[time_col] >= self.video_start_fiber) & 
                    (self.fiber_data[time_col] <= self.video_end_fiber)]
                
                if not fiber_in_range.empty:
                    fiber_rel_time = to_relative_time(fiber_in_range[time_col])
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.plot(fiber_rel_time, fiber_in_range['normalized_signal'], 
                            linewidth=1.5, color='purple', label='Fiber Signal')
                    ax3.set_ylabel("Normalized Fiber Signal", color='purple')
                    ax3.tick_params(axis='y', labelcolor='purple')

            plt.title("Pupil Distance, Speed and Fiber Signal Timeline")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if 'ax3' in locals():
                lines3, labels3 = ax3.get_legend_handles_labels()
            else:
                lines3, labels3 = [], []
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#b6b6b6ff", alpha=0.3, label='Wait'),
                Patch(facecolor="#fcb500f0", alpha=0.3, label='Stimulation')
            ]
            
            all_lines = lines1 + lines2 + lines3 + legend_elements
            all_labels = labels1 + labels2 + labels3 + ['Wait', 'Stimulation']
            
            ax1.legend(handles=all_lines, labels=all_labels, 
                    loc='upper right', bbox_to_anchor=(1.3, 1))

            plt.tight_layout()
            plt.show()
            
            self.set_status("Combined plot completed.")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot combined data: {str(e)}")
            self.set_status("Plot failed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FreezingAnalyzerApp(root)
    root.mainloop()