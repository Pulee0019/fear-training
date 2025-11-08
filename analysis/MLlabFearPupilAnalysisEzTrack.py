import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import struct
import re
import os
from datetime import datetime
from itertools import groupby
import matplotlib.dates as mdates

invert = True

class FreezingAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Conditioning Analyzer")
        self.root.geometry("200x500")
        self.root.resizable(False, False)
        
        self.eztrack_results_path = ""
        self.dlc_results_path = ""
        self.timestamp_path = ""
        self.ast2_path = ""
        self.events_path = ""
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
        
        self.freezing_frame = ttk.LabelFrame(main_frame, text="Freezing Analysis")
        self.freezing_frame.grid(row=2, column=0, pady=10, sticky="EW")
        
        self.create_button(self.freezing_frame, "Load ezTrack Results", self.load_eztrack_Results, 0)
        self.create_button(self.freezing_frame, "Load Events CSV", self.load_events, 1)
        self.create_button(self.freezing_frame, "Run Freezing Analysis", self.analyze, 2)
        self.create_button(self.freezing_frame, "Plot Freezing Timeline", self.plot_freezing_windows, 3)
        self.create_button(self.freezing_frame, "Export Results to CSV", self.export_freezing_csv, 4)
        
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
            
            self.freezing_data = self.compute_freezing_from_eztrack()
            
            self.compute_freezing_by_window(window_size_sec=30, fps=30)
            
            messagebox.showinfo("Done", "Freezing analysis completed!")
            self.set_status("Freezing analysis completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.set_status("Freezing analysis failed.")

    def compute_freezing_from_eztrack(self):
        freezing_percentages = {}
        
        for idx, event in self.event_data.iterrows():
            event_start = event['start_time']*30
            event_end = event['end_time']*30
            
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
        
        num_windows = total_frames // window_size_frames
        
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

            plt.title("Pupil Distance, Speed and Event Timeline (Relative to First Event)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#b6b6b6ff", alpha=0.3, label='Wait'),
                Patch(facecolor="#fcb500f0", alpha=0.3, label='Stimulation')
            ]
            
            ax1.legend(handles=lines1 + lines2 + legend_elements, 
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