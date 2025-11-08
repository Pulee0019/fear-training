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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

invert = True

class ModeSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Lab Fear Conditioning Analyzer")
        self.root.geometry("500x320")
        self.root.resizable(False, False)
        
        self.root.configure(bg="#f0f0f0")
        
        main_frame = ttk.Frame(root, padding=40)
        main_frame.pack(expand=True, fill="both")
        
        title_label = ttk.Label(main_frame, text="ML Lab Fear Conditioning Analyzer", 
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
        
        version_label = ttk.Label(main_frame, text="Version 3.0 | © 2025 ML Lab", 
                                 font=("Arial", 8))
        version_label.pack(side="bottom", pady=5)
        
    def select_mode(self, mode):
        response = messagebox.askyesno("Fiber Photometry", 
                                      "Do you want to include fiber photometry analysis?")
        
        self.root.destroy()
        
        root = tk.Tk()
        root.geometry("1200x1200")
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
        self.event_time_absolute = None
        self.channel_data = {}
        self.active_channels = []
        self.smooth_window = tk.IntVar(value=11)
        self.smooth_order = tk.IntVar(value=5)
        self.baseline_start = tk.DoubleVar(value=0)
        self.baseline_end = tk.DoubleVar(value=120)
        self.apply_smooth = tk.BooleanVar(value=False)
        self.apply_baseline = tk.BooleanVar(value=False)
        self.apply_motion = tk.BooleanVar(value=False)
        self.target_signal_var = tk.StringVar(value="470")
        self.reference_signal_var = tk.StringVar(value="410")
        self.baseline_period = [0, 120]
        self.baseline_model = tk.StringVar(value="Polynomial")  # New: baseline model selection
        self.multi_animal_data = []  # Store data for multiple animals
        self.current_animal_index = 0  # Track current animal in multi-animal mode
        
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
                    'eztrack': ['*freezingoutput.csv', '*eztrack*.csv'],
                    'events' : ['*events*.csv', '*timeline*.csv']
                }
            
            elif exp_mode == "freezing+fiber":
                required_files = {
                    'eztrack': ['*freezingoutput.csv', '*eztrack*.csv'],
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
                print(self.eztrack_results_path)
                self.set_status(f"Loaded ezTrack: {os.path.basename(self.eztrack_results_path)}")
                imported_files.append("ezTrack results")
            
            if 'dlc' in files_found:
                self.dlc_results_path = files_found['dlc']
                print(self.dlc_results_path)
                filename = os.path.basename(self.dlc_results_path)
                match = re.search(r'cam(\d+)', filename)
                if match:
                    self.cam_id = int(match.group(1))
                self.set_status(f"Loaded DLC: Camera {self.cam_id}")
                imported_files.append("DLC results")
            
            if 'events' in files_found:
                self.events_path = files_found['events']
                print(self.events_path)
                try:
                    self.event_data = pd.read_csv(self.events_path)
                    start_offset = self.event_data['start_time'].iloc[0]

                    self.event_data['start_time'] = self.event_data['start_time'] - start_offset
                    self.event_data['end_time'] = self.event_data['end_time'] - start_offset

                    self.event_time_absolute = False
                    imported_files.append("Events file")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load events: {str(e)}")
            
            if 'timestamp' in files_found:
                self.timestamp_path = files_found['timestamp']
                print(self.timestamp_path)
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
                print(self.ast2_path)
                try:
                    self.ast2_data = self.h_AST2_readData(self.ast2_path)
                    imported_files.append("AST2 file")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load AST2: {str(e)}")
            
            if 'fiber' in files_found:
                self.fiber_data_path = files_found['fiber']
                print(self.fiber_data_path)
                try:
                    self.load_fiber_data()
                    imported_files.append("Fiber data")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load fiber: {str(e)}")
            
            if self.include_fiber and self.fiber_data is not None:
                if len(self.channel_data) > 1:
                    self.show_channel_selection_dialog()
                else:
                    self.active_channels = list(self.channel_data.keys())
                    self.set_status(f"Auto-selected channel: CH{self.active_channels[0]}")
                    self.set_video_markers()  # Set video markers after auto-selecting channel

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

    def import_multi_animal_data(self):
        base_dir = filedialog.askdirectory(title="Select Free Moving/Behavioural directory")
        if not base_dir:
            return

        exp_mode = "freezing+fiber" if self.include_fiber else "freezing"
        if "PupilAnalyzerApp" in str(type(self)):
            exp_mode = "pupil+fiber" if self.include_fiber else "pupil"

        try:
            self.set_status("Scanning for multi-animal data...")
            self.selected_files = []
            self.multi_animal_data = []
            self.file_listbox.delete(0, tk.END)

            batch_dirs = glob.glob(os.path.join(base_dir, "20*"))
            for batch_dir in batch_dirs:
                if not os.path.isdir(batch_dir):
                    continue

                batch_name = os.path.basename(batch_dir)

                for group in ["CFC", "EXT", "RET", "CFT"]:
                    group_dir = os.path.join(batch_dir, group)
                    if not os.path.exists(group_dir):
                        continue

                    ear_tag_dirs = glob.glob(os.path.join(group_dir, "*"))
                    for ear_tag_dir in ear_tag_dirs:
                        if not os.path.isdir(ear_tag_dir):
                            continue

                        ear_tag = os.path.basename(ear_tag_dir)
                        animal_id = f"{batch_name}-{ear_tag}"

                        files_found = {}
                        patterns = {
                            'eztrack': ['*freezingoutput.csv', '*eztrack*.csv'],
                            'events': ['*events*.csv', '*timeline*.csv'],
                            'fiber': ['fluorescence.csv', '*fiber*.csv'],
                            'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
                            'timestamp': ['*timestamp*.csv', '*time*.csv'],
                            'ast2': ['*.ast2']
                        }

                        for file_type, file_patterns in patterns.items():
                            found_file = None
                            for root_path, dirs, files in os.walk(ear_tag_dir):
                                for file in files:
                                    file_lower = file.lower()
                                    for pattern in file_patterns:
                                        if fnmatch.fnmatch(file_lower, pattern.lower()):
                                            found_file = os.path.join(root_path, file)
                                            files_found[file_type] = found_file
                                            break
                                    if found_file:
                                        break
                                if found_file:
                                    break

                        animal_data = {
                            'animal_id': animal_id,
                            'group': group,
                            'files': files_found,
                            'processed': False,
                            'event_time_absolute': False,
                            'active_channels': []
                        }

                        required_files = ['eztrack'] if 'freezing' in exp_mode else ['dlc', 'timestamp']
                        if all(ft in files_found for ft in required_files):
                            self.selected_files.append(animal_data)
                            self.file_listbox.insert(tk.END, f"{animal_id} ({group})")

                            # Process events file
                            if 'events' in files_found:
                                try:
                                    event_data = pd.read_csv(files_found['events'])
                                    start_offset = event_data['start_time'].iloc[0]
                                    event_data['start_time'] = event_data['start_time'] - start_offset
                                    event_data['end_time'] = event_data['end_time'] - start_offset
                                    animal_data['event_data'] = event_data
                                except Exception as e:
                                    print(f"Failed to load events for {animal_id}: {str(e)}")

                            # Process ezTrack file
                            if 'eztrack' in files_found:
                                try:
                                    eztrack_result = pd.read_csv(files_found['eztrack'])
                                    animal_data['eztrack_result'] = eztrack_result
                                except Exception as e:
                                    print(f"Failed to load ezTrack for {animal_id}: {str(e)}")

                            # Process DLC file
                            if 'dlc' in files_found:
                                try:
                                    dlc_data = pd.read_csv(files_found['dlc'])
                                    animal_data['dlc_data'] = dlc_data
                                    filename = os.path.basename(files_found['dlc'])
                                    match = re.search(r'cam(\d+)', filename)
                                    if match:
                                        animal_data['cam_id'] = int(match.group(1))
                                except Exception as e:
                                    print(f"Failed to load DLC for {animal_id}: {str(e)}")

                            # Process timestamp file
                            if 'timestamp' in files_found:
                                try:
                                    timestamps = pd.read_csv(files_found['timestamp'])
                                    exp_start = timestamps[(timestamps['Device'] == 'Experiment') &
                                                        (timestamps['Action'] == 'Start')]['Timestamp'].values
                                    if len(exp_start) > 0:
                                        animal_data['experiment_start'] = exp_start[0]
                                except Exception as e:
                                    print(f"Failed to load timestamp for {animal_id}: {str(e)}")

                            # Process AST2 file
                            if 'ast2' in files_found:
                                try:
                                    ast2_data = self.h_AST2_readData(files_found['ast2'])
                                    animal_data['ast2_data'] = ast2_data
                                except Exception as e:
                                    print(f"Failed to load AST2 for {animal_id}: {str(e)}")

                            # Process fiber data
                            if 'fiber' in files_found:
                                try:
                                    fiber_result = self.load_fiber_data(files_found['fiber'])
                                    if fiber_result:
                                        animal_data.update(fiber_result)
                                except Exception as e:
                                    print(f"Failed to load fiber for {animal_id}: {str(e)}")

                            animal_data['processed'] = True
                            self.multi_animal_data.append(animal_data)

            if not self.selected_files:
                messagebox.showwarning("No Data", "No valid animal data found in the selected directory")
                self.set_status("No multi-animal data found")
            else:
                self.set_status(f"Found {len(self.selected_files)} animals")
                # Show channel selection dialog for fiber data
                if self.include_fiber:
                    self.show_channel_selection_dialog()
                else:
                    messagebox.showinfo("Success", f"Found and processed {len(self.selected_files)} animals")
                    self.set_status(f"Imported {len(self.selected_files)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            self.set_status("Import failed")

    def load_fiber_data(self, file_path=None):
        path = file_path or self.fiber_data_path
        try:
            self.fiber_data = pd.read_csv(path, skiprows=1, delimiter=',')
            self.fiber_data = self.fiber_data.loc[:, ~self.fiber_data.columns.str.contains('^Unnamed')]
            # print(f"Columns in fiber data: {fiber_data.columns}")
            self.fiber_data.columns = self.fiber_data.columns.str.strip()
            # print(f"Columns after stripping: {fiber_data.columns}")

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
            
            return {
                'fiber_data': self.fiber_data,
                'channels': self.channels,
                'channel_data': self.channel_data
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fiber data: {str(e)}")
            self.set_status("Fiber data load failed")
            return None
    
    def show_channel_selection_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Channels")
        dialog.geometry("500x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.channel_vars = {}
        
        if self.multi_animal_data:
            for animal_data in self.multi_animal_data:
                if 'channel_data' not in animal_data:
                    continue
                    
                animal_id = animal_data['animal_id']
                group = animal_data.get('group', '')
                label = f"{animal_id} ({group})" if group else animal_id
                
                animal_frame = ttk.LabelFrame(scrollable_frame, text=label)
                animal_frame.pack(fill="x", padx=5, pady=5, ipadx=5, ipady=5)
                
                self.channel_vars[animal_id] = {}
                
                for channel_num in sorted(animal_data['channel_data'].keys()):
                    var = tk.BooleanVar(value=(channel_num == 1))
                    self.channel_vars[animal_id][channel_num] = var
                    
                    chk = ttk.Checkbutton(
                        animal_frame, 
                        text=f"Channel {channel_num}",
                        variable=var
                    )
                    chk.pack(anchor="w", padx=20, pady=2)
        else:
            if hasattr(self, 'channel_data') and self.channel_data:
                animal_frame = ttk.LabelFrame(scrollable_frame, text="Single Animal")
                animal_frame.pack(fill="x", padx=5, pady=5, ipadx=5, ipady=5)
                
                self.channel_vars['single'] = {}
                
                for channel_num in sorted(self.channel_data.keys()):
                    var = tk.BooleanVar(value=(channel_num == 1))
                    self.channel_vars['single'][channel_num] = var
                    
                    chk = ttk.Checkbutton(
                        animal_frame, 
                        text=f"Channel {channel_num}",
                        variable=var
                    )
                    chk.pack(anchor="w", padx=20, pady=2)
        
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Select All", command=lambda: self.toggle_all_channels(True)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: self.toggle_all_channels(False)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Confirm", command=lambda: self.finalize_channel_selection(dialog)).pack(side="right", padx=5)
        
    def toggle_all_channels(self, select):
        for animal_vars in self.channel_vars.values():
            for var in animal_vars.values():
                var.set(select)

    def finalize_channel_selection(self, dialog):
        if self.multi_animal_data:
            for animal_data in self.multi_animal_data:
                animal_id = animal_data['animal_id']
                
                if animal_id in self.channel_vars:
                    selected_channels = []
                    for channel_num, var in self.channel_vars[animal_id].items():
                        if var.get():
                            selected_channels.append(channel_num)
                    
                    if not selected_channels:
                        messagebox.showwarning(
                            "No Channels Selected", 
                            f"Please select at least one channel for {animal_id}"
                        )
                        return
                        
                    animal_data['active_channels'] = selected_channels
                    self.set_video_markers(animal_data)
        else:
            selected_channels = []
            for channel_num, var in self.channel_vars['single'].items():
                if var.get():
                    selected_channels.append(channel_num)
            
            if not selected_channels:
                messagebox.showwarning(
                    "No Channels Selected", 
                    "Please select at least one channel"
                )
                return
            
            self.active_channels = selected_channels
            self.set_video_markers()

        dialog.destroy()
        if self.multi_animal_data:
            self.set_status(f"Selected channels for all animals")
            messagebox.showinfo("Success", f"Processed {len(self.multi_animal_data)} animals")
            self.set_status(f"Imported {len(self.multi_animal_data)} animals")
        else:
            self.set_status("Selected channels for single animal")
            messagebox.showinfo("Success", "Processed single animal")
    
    def set_video_markers(self, animal_data=None):
        """Set video markers for either single animal or specific animal in multi-animal mode"""
        try:
            # Determine which data to use
            if animal_data:
                fiber_data = animal_data.get('fiber_data')
                event_data = animal_data.get('event_data')
                channels = animal_data.get('channels', {})
                active_channels = animal_data.get('active_channels', [])
            else:
                fiber_data = self.fiber_data
                event_data = self.event_data
                channels = self.channels
                active_channels = self.active_channels
            
            # Check if we have the necessary data
            if fiber_data is None or not active_channels:
                messagebox.showwarning("No Data", "Please load fiber data and select channels first")
                return
            
            events_col = channels.get('events')
            if events_col is None or events_col not in fiber_data.columns:
                messagebox.showerror("Error", "Events column not found in fiber data")
                return
            
            input1_events = fiber_data[fiber_data[events_col].str.contains('Input1', na=False)]
            if len(input1_events) < 2:
                messagebox.showerror("Error", "Could not find enough Input1 events (need at least 2)")
                return
            
            time_col = channels['time']
            video_start_fiber = input1_events[time_col].iloc[0]
            video_end_fiber = input1_events[time_col].iloc[-1]
            
            fiber_cropped = fiber_data[
                (fiber_data[time_col] >= video_start_fiber) & 
                (fiber_data[time_col] <= video_end_fiber)].copy()
            
            # Update the appropriate data structure
            if animal_data:
                animal_data.update({
                    'fiber_cropped': fiber_cropped,
                    'video_start_fiber': video_start_fiber,
                    'video_end_fiber': video_end_fiber
                })
            else:
                self.fiber_cropped = fiber_cropped
                self.video_start_fiber = video_start_fiber
                self.video_end_fiber = video_end_fiber
            
            # Convert event times if event data exists
            if event_data is not None and not (animal_data.get('event_time_absolute') if animal_data else self.event_time_absolute):
                # Convert to absolute time
                event_data['start_absolute'] = video_start_fiber + event_data['start_time']
                event_data['end_absolute'] = video_start_fiber + event_data['end_time']
                
                if animal_data:
                    animal_data['event_time_absolute'] = True
                else:
                    self.event_time_absolute = True
            
            messagebox.showinfo("Success", 
                               f"Video markers set:\nStart: {video_start_fiber:.2f}s\nEnd: {video_end_fiber:.2f}s")
            self.set_status("Video markers set from fiber data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set video markers: {str(e)}")
            self.set_status("Video markers failed")
    
    def plot_raw_data(self):
        if not self.multi_animal_data and (self.fiber_cropped is None or not self.active_channels):
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            if self.multi_animal_data:
                # Group animals by group
                groups = {}
                for animal_data in self.multi_animal_data:
                    group = animal_data.get('group', 'Unknown')
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(animal_data)
                # Create a new window for multi-animal plots
                plot_window = tk.Toplevel(self.root)
                plot_window.title("Multi-Animal Raw Data")
                plot_window.geometry("1000x800")
                
                # Create tabs for each group
                tab_control = ttk.Notebook(plot_window)
                
                for group_name, animal_list in groups.items():
                    tab = ttk.Frame(tab_control)
                    tab_control.add(tab, text=group_name)
                    
                    fig = Figure(figsize=(10, 6), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    
                    colors = plt.cm.tab10(np.linspace(0, 1, len(animal_list)))
                    for animal_idx, animal_data in enumerate(animal_list):
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['fiber_cropped'][time_col] - animal_data['video_start_fiber']
                        
                        for channel_num in animal_data['active_channels']:
                            if channel_num in animal_data['channel_data']:
                                for wavelength, col_name in animal_data['channel_data'][channel_num].items():
                                    if col_name and col_name in animal_data['fiber_cropped'].columns:
                                        animal_id = animal_data['animal_id']
                                        base_color = colors[animal_idx]
                                        alpha = 1.0 if wavelength == self.target_signal_var.get() else 0.5
                                        ax.plot(time_data, animal_data['fiber_cropped'][col_name], 
                                            color=base_color, linewidth=1, alpha=alpha, 
                                            label=f'{animal_id} CH{channel_num} {wavelength}nm')
                    
                    ax.set_title(f"Group: {group_name}")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Signal Intensity")
                    ax.grid(False)
                    
                    # Add events if available
                    event_data = animal_data.get('event_data')
                    if event_data is not None:
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax.axvspan(start_time, end_time, color=color, alpha=0.3)
                    
                    ax.legend()
                    fig.tight_layout()
                    canvas.draw()
                
                tab_control.pack(expand=1, fill="both")
                self.set_status("Raw fiber data plotted in new window")
            else:
                # Single animal mode
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                
                time_col = self.channels['time']
                time_data = self.fiber_cropped[time_col] - self.video_start_fiber
                
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                for i, channel_num in enumerate(self.active_channels):
                    if channel_num in self.channel_data:
                        for wavelength, col_name in self.channel_data[channel_num].items():
                            if col_name and col_name in self.fiber_cropped.columns:
                                base_color = colors[i % len(colors)]
                                alpha = 1.0 if wavelength == self.target_signal_var.get() else 0.5
                                ax.plot(time_data, self.fiber_cropped[col_name], 
                                    color=base_color, linewidth=1, alpha=alpha, 
                                    label=f'CH{channel_num} {wavelength}nm')
                
                ax.set_title("Raw Fiber Photometry Data")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Signal Intensity")
                ax.grid(False)
                
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
                        start_time = row[start_col] - self.video_start_fiber
                        end_time = row[end_col] - self.video_start_fiber
                        ax.axvspan(start_time, end_time, color=color, alpha=0.3)
                
                ax.legend()
                self.fig.tight_layout()
                self.canvas.draw()
                self.set_status("Raw fiber data plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            self.set_status("Raw plot failed")
    
    def smooth_data(self):
        if not self.multi_animal_data and (self.fiber_cropped is None or not self.active_channels):
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            window_size = self.smooth_window.get()
            poly_order = self.smooth_order.get()
            
            if window_size % 2 == 0:
                window_size += 1
            
            if self.multi_animal_data:
                # Apply smoothing to all animals in multi-animal mode
                for animal_data in self.multi_animal_data:
                    animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                            if target_col and target_col in animal_data['preprocessed_data'].columns:
                                smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                                animal_data['preprocessed_data'][smoothed_col] = savgol_filter(
                                    animal_data['preprocessed_data'][target_col], window_size, poly_order)
            else:
                # Single animal mode
                self.preprocessed_data = self.fiber_cropped.copy()
                for channel_num in self.active_channels:
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in self.preprocessed_data.columns:
                            smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            self.preprocessed_data[smoothed_col] = savgol_filter(
                                self.preprocessed_data[target_col], window_size, poly_order)
            
            fig, ax = plt.subplots()
            
            if self.multi_animal_data:
                # Plot smoothed data for all animals
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.multi_animal_data)))
                for idx, animal_data in enumerate(self.multi_animal_data):
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                            smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            
                            if target_col and target_col in animal_data['preprocessed_data'].columns:
                                animal_id = animal_data['animal_id']
                                ax.plot(time_data, animal_data['preprocessed_data'][target_col], 
                                        color=colors[idx], linestyle='-', alpha=0.5, label=f'{animal_id} Raw')

                            if smoothed_col in animal_data['preprocessed_data'].columns:
                                animal_id = animal_data['animal_id']
                                ax.plot(time_data, animal_data['preprocessed_data'][smoothed_col], 
                                        color=colors[idx], linestyle='-', linewidth=2, label=f'{animal_id} Smoothed')
            else:
                # Single animal mode
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
            plt.show()
            self.set_status(f"Data smoothed with window={window_size}, order={poly_order}")
        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed: {str(e)}")
            self.set_status("Smoothing failed")
    
    def baseline_correction(self):
        if not self.multi_animal_data and (self.fiber_cropped is None or not self.active_channels):
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            model_type = self.baseline_model.get()
            
            if self.multi_animal_data:
                # Apply to all animals in multi-animal mode
                for animal_data in self.multi_animal_data:
                    if animal_data['preprocessed_data'] is None:
                        animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                        
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    full_time = time_data.values
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                            if not target_col or target_col not in animal_data['preprocessed_data'].columns:
                                continue

                            if f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in animal_data['preprocessed_data'].columns:
                                signal_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            else:
                                signal_col = target_col

                            signal_data = animal_data['preprocessed_data'][signal_col].values
                            
                            if model_type.lower() == "exponential":
                                def exp_model(t, a, b, c):
                                    return a * np.exp(-b * t) + c
                                
                                p0 = [
                                    np.max(signal_data) - np.min(signal_data),
                                    0.01,
                                    np.min(signal_data)
                                ]
                                
                                try:
                                    params, _ = curve_fit(exp_model, full_time, signal_data, p0=p0, maxfev=5000)
                                    baseline_pred = exp_model(full_time, *params)
                                except Exception as e:
                                    self.set_status(f"Exponential fit failed: {str(e)}, using polynomial instead")
                                    X = full_time.reshape(-1, 1)
                                    model = LinearRegression()
                                    model.fit(X, signal_data)
                                    baseline_pred = model.predict(X)
                            else:
                                X = full_time.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(X, signal_data)
                                baseline_pred = model.predict(X)
                            
                            baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                            animal_data['preprocessed_data'][baseline_corrected_col] = signal_data - baseline_pred
                            
                            animal_data['preprocessed_data'][f"CH{channel_num}_baseline_pred"] = baseline_pred
            else:
                # Single animal mode
                if self.preprocessed_data is None:
                    self.preprocessed_data = self.fiber_cropped.copy()
                    
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                full_time = time_data.values
                
                for channel_num in self.active_channels:
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if not target_col or target_col not in self.preprocessed_data.columns:
                            continue
                        if f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in self.preprocessed_data.columns:
                            signal_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                        else:
                            signal_col = target_col

                        signal_data = self.preprocessed_data[signal_col].values
                        
                        if model_type.lower() == "exponential":
                            def exp_model(t, a, b, c):
                                return a * np.exp(-b * t) + c
                            
                            p0 = [
                                np.max(signal_data) - np.min(signal_data),
                                0.01,
                                np.min(signal_data)
                            ]
                            
                            try:
                                params, _ = curve_fit(exp_model, full_time, signal_data, p0=p0, maxfev=5000)
                                baseline_pred = exp_model(full_time, *params)
                            except Exception as e:
                                self.set_status(f"Exponential fit failed: {str(e)}, using polynomial instead")
                                X = full_time.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(X, signal_data)
                                baseline_pred = model.predict(X)
                        else:
                            X = full_time.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, signal_data)
                            baseline_pred = model.predict(X)
                        
                        baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                        self.preprocessed_data[baseline_corrected_col] = signal_data - baseline_pred
                        
                        self.preprocessed_data[f"CH{channel_num}_baseline_pred"] = baseline_pred
            
            fig, ax = plt.subplots()
            
            if self.multi_animal_data:
                # Plot baseline correction for all animals
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.multi_animal_data)))
                for idx, animal_data in enumerate(self.multi_animal_data):
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            baseline_corrected_col = f"CH{channel_num}_baseline_corrected"
                            if baseline_corrected_col in animal_data['preprocessed_data'].columns:
                                animal_id = animal_data['animal_id']
                                color = colors[idx]
                                
                                target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                                if target_col and target_col in animal_data['preprocessed_data'].columns:
                                    ax.plot(time_data, animal_data['preprocessed_data'][target_col], 
                                        color=color, linestyle='-', alpha=0.5, label=f'{animal_id} Original')
                                
                                ax.plot(time_data, animal_data['preprocessed_data'][f"CH{channel_num}_baseline_pred"],
                                    color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f'{animal_id} Baseline Fit')
                                
                                ax.plot(time_data, animal_data['preprocessed_data'][baseline_corrected_col], 
                                    color=color, linestyle='-', linewidth=2, label=f'{animal_id} Corrected')
            else:
                # Single animal mode
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                for i, channel_num in enumerate(self.active_channels):
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in self.preprocessed_data.columns:
                            ax.plot(time_data, self.preprocessed_data[target_col], 
                                'b-', alpha=0.5, label=f'Original')
                        
                        ax.plot(time_data, self.preprocessed_data[f"CH{channel_num}_baseline_pred"],
                            'g--', linewidth=1.5, alpha=0.7, label=f'Baseline Fit')
                        
                        ax.plot(time_data, self.preprocessed_data[f"CH{channel_num}_baseline_corrected"], 
                            'r-', label=f'Corrected')
            
            ax.set_title(f"Baseline Correction ({model_type} Model)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            ax.legend()
            plt.show()
            self.set_status(f"Baseline correction applied ({model_type} model) for bleaching effect")
        except Exception as e:
            messagebox.showerror("Error", f"Baseline correction failed: {str(e)}")
            self.set_status("Baseline correction failed")
    
    def motion_correction(self):
        if self.reference_signal_var.get() != "410":
            messagebox.showwarning("Invalid Reference", "Motion correction requires 410nm as reference signal")
            return
            
        if not self.multi_animal_data and (self.fiber_cropped is None or not self.active_channels):
            messagebox.showwarning("No Data", "Please load and crop fiber data and select channels first")
            return
        
        try:
            if self.multi_animal_data:
                # Apply to all animals in multi-animal mode
                for animal_data in self.multi_animal_data:
                    if animal_data['preprocessed_data'] is None:
                        animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                        
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            ref_col = animal_data['channel_data'][channel_num].get('410')
                            if not ref_col or ref_col not in animal_data['preprocessed_data'].columns:
                                self.set_status(f"No 410nm data for channel CH{channel_num}")
                                continue
                            
                            target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                            if not target_col or target_col not in animal_data['preprocessed_data'].columns:
                                continue
                            
                            if f"CH{channel_num}_baseline_corrected" in animal_data['preprocessed_data'].columns:
                                signal_col = f"CH{channel_num}_baseline_corrected"
                            elif f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in animal_data['preprocessed_data'].columns:
                                signal_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            else:
                                signal_col = target_col
                            
                            signal_data = animal_data['preprocessed_data'][signal_col]
                            ref_data = animal_data['preprocessed_data'][ref_col]
                            
                            X = ref_data.values.reshape(-1, 1)
                            y = signal_data.values
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            predicted_signal = model.predict(X)
                            
                            motion_corrected_col = f"CH{channel_num}_motion_corrected"
                            animal_data['preprocessed_data'][motion_corrected_col] = signal_data - predicted_signal
                            # Save fitted reference signal
                            fitted_ref_col = f"CH{channel_num}_fitted_ref"
                            animal_data['preprocessed_data'][fitted_ref_col] = predicted_signal
            else:
                # Single animal mode
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
                        # Save fitted reference signal
                        fitted_ref_col = f"CH{channel_num}_fitted_ref"
                        self.preprocessed_data[fitted_ref_col] = predicted_signal
            
            fig, ax = plt.subplots()
            
            if self.multi_animal_data:
                # Plot motion correction for all animals
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.multi_animal_data)))
                for idx, animal_data in enumerate(self.multi_animal_data):
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            motion_corrected_col = f"CH{channel_num}_motion_corrected"
                            if motion_corrected_col in animal_data['preprocessed_data'].columns:
                                animal_id = animal_data['animal_id']
                                color = colors[idx]
                                ax.plot(time_data, animal_data['preprocessed_data'][motion_corrected_col], 
                                    color=color, linestyle='-', label=f'{animal_id} Motion Corrected')
            else:
                # Single animal mode
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
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
            plt.show()
            self.set_status("Motion correction applied")
        except Exception as e:
            messagebox.showerror("Error", f"Motion correction failed: {str(e)}")
            self.set_status("Motion correction failed")
    
    def calculate_and_plot_dff(self):
        if not self.multi_animal_data and (self.preprocessed_data is None or not self.active_channels):
            messagebox.showwarning("No Data", "Please preprocess data and select channels first")
            return
        
        try:
            if self.multi_animal_data:
                # Calculate dF/F for all animals
                for animal_data in self.multi_animal_data:
                    if animal_data['preprocessed_data'] is None:
                        animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()

                    animal_data['dff_data'] = {}
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                    
                    if not any(baseline_mask):
                        messagebox.showerror("Error", "No data in baseline period")
                        return
                        
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            ref_wavelength = self.reference_signal_var.get()
                            ref_col = animal_data['channel_data'][channel_num].get(ref_wavelength)
                            # Get raw target signal
                            target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                            if not target_col or target_col not in animal_data['preprocessed_data'].columns:
                                continue
                            
                            if f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in animal_data['preprocessed_data'].columns:
                                raw_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            else:
                                raw_col = target_col
                            
                            raw_target = animal_data['preprocessed_data'][raw_col]
                            median_full = np.median(raw_target)  # median(Ffull)
                            
                            # Case 1: Reference is 410 with baseline correction
                            if ref_wavelength == "410" and self.apply_baseline.get():
                                # MotionCorrected fitted410
                                if f"CH{channel_num}_motion_corrected" in animal_data['preprocessed_data'].columns:
                                    motion_corrected = animal_data['preprocessed_data'][f"CH{channel_num}_motion_corrected"]
                                    dff_data = motion_corrected / median_full
                                else:
                                    messagebox.showerror("Error", "Please apply the motion correction")
                            
                            # Case 2: Reference is 410 without baseline correction
                            elif ref_wavelength == "410" and not self.apply_baseline.get():
                                # MotionCorrected fitted410
                                if f"CH{channel_num}_fitted_ref" in animal_data['preprocessed_data'].columns:
                                    fitted_ref = animal_data['preprocessed_data'][f"CH{channel_num}_fitted_ref"]
                                    dff_data = (raw_target - fitted_ref) / fitted_ref
                                else:
                                    messagebox.showerror("Error", "Please apply the motion correction")
                            
                            # Case 3: Reference is baseline with baseline correction
                            elif ref_wavelength == "baseline" and self.apply_baseline.get():
                                if f"CH{channel_num}_baseline_pred" in animal_data['preprocessed_data'].columns:
                                    baseline_fitted = animal_data['preprocessed_data'][f"CH{channel_num}_baseline_pred"]
                                    baseline_median = np.median(raw_target[baseline_mask])
                                    dff_data = (raw_target - baseline_fitted) / baseline_median
                                else:
                                    messagebox.showerror("Error", "Please check the baseline correction")
                            
                            # Case 4: Reference is baseline without baseline correction
                            elif ref_wavelength == "baseline" and not self.apply_baseline.get():
                                baseline_median = np.median(raw_target[baseline_mask])
                                dff_data = (raw_target - baseline_median) / baseline_median
                            
                            dff_col = f"CH{channel_num}_dff"
                            animal_data['preprocessed_data'][dff_col] = dff_data
                            animal_data['dff_data'][channel_num] = dff_data
            else:
                # Single animal mode
                if self.preprocessed_data is None:
                    self.preprocessed_data = self.fiber_cropped.copy()

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
                        ref_wavelength = self.reference_signal_var.get()
                        # Get raw target signal
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if not target_col or target_col not in self.preprocessed_data.columns:
                            continue
                        if f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in self.preprocessed_data.columns:
                            raw_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                        else:
                            raw_col = target_col

                        raw_target = self.preprocessed_data[raw_col]
                        median_full = np.median(raw_target)  # median(Ffull)
                        
                        # Case 1: Reference is 410 with baseline correction
                        if ref_wavelength == "410" and self.apply_baseline.get():
                            # MotionCorrected fitted410
                            if f"CH{channel_num}_motion_corrected" in self.preprocessed_data.columns:
                                motion_corrected = self.preprocessed_data[f"CH{channel_num}_motion_corrected"]
                                dff_data = motion_corrected / median_full
                            else:
                                messagebox.showerror("Error", "Please apply the motion correction")
                        
                        # Case 2: Reference is 410 without baseline correction
                        elif ref_wavelength == "410" and not self.apply_baseline.get():
                            # MotionCorrected fitted410
                            if f"CH{channel_num}_fitted_ref" in self.preprocessed_data.columns:
                                fitted_ref = self.preprocessed_data[f"CH{channel_num}_fitted_ref"]
                                dff_data = (raw_target - fitted_ref) / fitted_ref
                            else:
                                messagebox.showerror("Error", "Please apply the motion correction")
                        
                        # Case 3: Reference is baseline with baseline correction
                        elif ref_wavelength == "baseline" and self.apply_baseline.get():
                            if f"CH{channel_num}_baseline_pred" in self.preprocessed_data.columns:
                                baseline_fitted = self.preprocessed_data[f"CH{channel_num}_baseline_pred"]
                                baseline_median = np.median(raw_target[baseline_mask])
                                dff_data = (raw_target - baseline_fitted) / baseline_median
                            else:
                                messagebox.showerror("Error", "Please check the baseline correction")
                        
                        # Case 4: Reference is baseline without baseline correction
                        elif ref_wavelength == "baseline" and not self.apply_baseline.get():
                            baseline_median = np.median(raw_target[baseline_mask])
                            dff_data = (raw_target - baseline_median) / baseline_median
                        
                        dff_col = f"CH{channel_num}_dff"
                        self.preprocessed_data[dff_col] = dff_data
                        self.dff_data[channel_num] = dff_data
                        
                        color = colors[i % len(colors)]
                        ax.plot(time_data, dff_data, f'{color}-', label=f'CH{channel_num} ΔF/F')

                if self.apply_baseline.get() and ref_wavelength == "baseline":
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
            if self.multi_animal_data:
                # Calculate z-score for all animals
                for animal_data in self.multi_animal_data:
                    if animal_data['dff_data'] is None or not animal_data['active_channels']:
                        messagebox.showwarning("No Data", "Please calculate ΔF/F first")
                        return
                        
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                    
                    if not any(baseline_mask):
                        messagebox.showerror("Error", "No data in baseline period")
                        return
                        
                    animal_data['zscore_data'] = {}
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['dff_data']:
                            dff_data = animal_data['dff_data'][channel_num]
                            baseline_dff = dff_data[baseline_mask]
                            
                            if len(baseline_dff) < 2:
                                continue
                            
                            mean_dff = np.mean(baseline_dff)
                            std_dff = np.std(baseline_dff)
                            
                            zscore_data = (dff_data - mean_dff) / std_dff
                            
                            zscore_col = f"CH{channel_num}_zscore"
                            animal_data['preprocessed_data'][zscore_col] = zscore_data
                            animal_data['zscore_data'][channel_num] = zscore_data
            else:
                # Single animal mode
                if self.dff_data is None or not self.active_channels:
                    messagebox.showwarning("No Data", "Please calculate ΔF/F first")
                    return
                    
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                ref_wavelength = self.reference_signal_var.get()
                baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                
                if not any(baseline_mask):
                    messagebox.showerror("Error", "No data in baseline period")
                    return
                    
                self.fig.clear()
                ax = self.fig.add_subplot(111)

                self.zscore_data = {}
                
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
                        self.zscore_data[channel_num] = zscore_data
                        
                        color = colors[i % len(colors)]
                        ax.plot(time_data, zscore_data, f'{color}-', label=f'CH{channel_num} Z-score')
                
                if self.apply_baseline.get() and ref_wavelength == "baseline":
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
    
    def select_event_occurrences_dialog(self, events, group_name):
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Select Event Occurrences - {group_name}")
        dialog.geometry("400x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text=f"Select which event occurrences to analyze for {group_name}:").pack(pady=5)
        
        scroll_frame = ttk.Frame(main_frame)
        scroll_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(scroll_frame)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        event_vars = []
        
        for idx, (_, event) in enumerate(events.iterrows()):
            var = tk.BooleanVar(value=True)
            event_vars.append(var)
            
            start_time = event['start_absolute'] if self.event_time_absolute else event['start_time']
            end_time = event['end_absolute'] if self.event_time_absolute else event['end_time']
            
            event_info = f"Event {idx+1}: {start_time:.1f}s - {end_time:.1f}s"
            chk = ttk.Checkbutton(
                scrollable_frame, 
                text=event_info,
                variable=var
            )
            chk.pack(anchor="w", padx=10, pady=2)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Select All", 
                command=lambda: [var.set(True) for var in event_vars]).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Deselect All", 
                command=lambda: [var.set(False) for var in event_vars]).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Confirm", 
                command=dialog.destroy).pack(side="right", padx=5)
        
        dialog.wait_window()
        return [idx for idx, var in enumerate(event_vars) if var.get()]

    def finalize_event_occurrence_selection(self, dialog):
        self.selected_occurrences = []
        for idx, var in enumerate(self.event_occurrence_vars):
            if var.get():
                self.selected_occurrences.append(idx)
        dialog.destroy()
    
    def plot_event_related_activity(self):
        if self.event_data is None and (not self.multi_animal_data or not any('event_data' in a for a in self.multi_animal_data)):
            messagebox.showwarning("No Event Data", "Please load event data first")
            return

        if self.multi_animal_data and not all('zscore_data' in a for a in self.multi_animal_data):
            self.calculate_and_plot_zscore()

        if not self.multi_animal_data and self.zscore_data is None:
            self.calculate_and_plot_zscore()

        try:
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            selected_type = int(self.event_type.get())
            
            if self.multi_animal_data:
                groups = {}
                for animal_data in self.multi_animal_data:
                    group = animal_data.get('group', 'Unknown')
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(animal_data)
                
                plot_window = tk.Toplevel(self.root)
                plot_window.title("Multi-Animal Event-Related Activity")
                plot_window.geometry("1000x800")
                
                tab_control = ttk.Notebook(plot_window)
                
                for group_name, animal_list in groups.items():
                    group_events = []
                    for animal_data in animal_list:
                        if 'event_data' in animal_data:
                            events = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type]
                            if not events.empty:
                                group_events.append((animal_data, events))
                    
                    if not group_events:
                        continue
                    
                    _, ref_events = group_events[0]
                    selected_indices = self.select_event_occurrences_dialog(ref_events, group_name)
                    
                    if not selected_indices:
                        continue
                    
                    tab = ttk.Frame(tab_control)
                    tab_control.add(tab, text=group_name)
                    
                    fig = Figure(figsize=(10, 6), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    
                    event_responses = {}
                    event_duration = None
                    max_event_num = 0
                    
                    for animal_data, events in group_events:
                        events = events.iloc[selected_indices]
                        
                        if events.empty:
                            continue
                        
                        if event_duration is None:
                            if animal_data.get('event_time_absolute', False):
                                event_duration = (events.iloc[0]['end_absolute'] - events.iloc[0]['start_absolute'])
                            else:
                                event_duration = events.iloc[0]['end_time'] - events.iloc[0]['start_time']
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        for event_idx, (_, event) in enumerate(events.iterrows(), 1):
                            orig_idx = selected_indices[event_idx - 1]
                            if orig_idx not in event_responses:
                                event_responses[orig_idx] = []
                                
                            if orig_idx > max_event_num:
                                max_event_num = orig_idx
                                
                            if animal_data.get('event_time_absolute', False):
                                start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                end_time = event['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start_time = event['start_time']
                                end_time = event['end_time']

                            event_start = start_time - pre_window
                            event_end = end_time + post_window

                            for channel_num in animal_data['active_channels']:
                                if channel_num in animal_data['zscore_data']:
                                    zscore_data = animal_data['zscore_data'][channel_num]
                            
                                    event_mask = (time_data >= event_start) & (time_data <= event_end)
                                    event_time_relative = time_data[event_mask] - start_time
                                    event_zscore = zscore_data[event_mask]
                                    
                                    if len(event_time_relative) > 0:
                                        event_responses[orig_idx].append((event_time_relative, event_zscore))
                    
                    all_zscores = []
                    for event_idx in event_responses:
                        for times, zscores in event_responses[event_idx]:
                            all_zscores.extend(zscores)
                    data_range = max(all_zscores) - min(all_zscores) if all_zscores else 5
                    y_spacing = data_range * 0.6 
                    
                    if event_responses:
                        colors = plt.cm.tab10(np.linspace(0, 1, max_event_num + 1))
                        
                        for event_idx in sorted(event_responses.keys()):
                            responses = event_responses[event_idx]
                            if not responses:
                                continue
                                
                            reference_times = None
                            max_length = max(len(t) for t, _ in responses)
                            
                            interpolated_responses = []
                            for times, zscores in responses:
                                if reference_times is None:
                                    reference_times = times
                                else:
                                    if len(times) != len(reference_times):
                                        zscores = np.interp(reference_times, times, zscores)
                                interpolated_responses.append(zscores)
                            
                            interpolated_responses = np.array(interpolated_responses)
                            mean_response = np.mean(interpolated_responses, axis=0)
                            std_response = np.std(interpolated_responses, axis=0, ddof=1)
                            sem_response = std_response / np.sqrt(interpolated_responses.shape[0]) 
                            
                            y_offset = (event_idx) * y_spacing
                            shifted_mean = mean_response + y_offset
                            shifted_lower = shifted_mean - sem_response
                            shifted_upper = shifted_mean + sem_response
                            
                            ax.plot(reference_times, shifted_mean, 
                                    color=colors[event_idx], 
                                    label=f'Event {event_idx + 1} (n={len(responses)} channels)')
                            ax.fill_between(reference_times, shifted_lower, shifted_upper, color=colors[event_idx], alpha=0.2)
                    
                    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Event Start')
                    if event_duration is not None:
                        ax.axvline(event_duration, color='k', linestyle='--', linewidth=1, label='Event End')
                        ax.axvspan(0, event_duration, color='yellow', alpha=0.2, label='Event Duration')

                    for animal_data in animal_list:
                        if 'event_data' in animal_data:
                            events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                            events1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 1]
                            for _, event2 in events2.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                    end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    start2 = event2['start_time']
                                    end2 = event2['end_time']

                                for _, event1 in events1.iterrows():
                                    if animal_data.get('event_time_absolute', False):
                                        event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                        event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                    else:
                                        event1_start = event1['start_time']
                                        event1_end = event1['end_time']

                                    if start2 >= event1_start and end2 <= event1_end:
                                        rel_start = start2 - event1_start
                                        ax.axvline(rel_start, color='r', linestyle='--', linewidth=1, 
                                                label='Type 2 Event' if 'Type 2 Event' not in ax.get_legend_handles_labels()[1] else "")
                    
                    ax.set_xlabel("Time Relative to Event Start (s)")
                    ax.set_ylabel("Z-Score")
                    ax.set_title(f"Group: {group_name}\nEvent-Related Activity by Event Occurrence ({len(animal_list)} animals, {len(selected_indices)} events)")
                    ax.grid(False)
                    ax.legend()
                        
                    canvas.draw()
                
                tab_control.pack(expand=1, fill="both")
                self.set_status("Multi-animal event-related activity plotted in new window")
            else:
                if self.zscore_data is None:
                    self.calculate_and_plot_zscore()
                
                if self.zscore_data is None:
                    return
                    
                events = self.event_data[self.event_data['Event Type'] == selected_type]
                if events.empty:
                    messagebox.showwarning("No Events", f"No events of type {selected_type} found")
                    return
                    
                selected_indices = self.select_event_occurrences_dialog(events, "Single Animal")
                if not selected_indices:
                    messagebox.showinfo("No Selection", "No events selected for analysis")
                    return
                    
                events = events.iloc[selected_indices]

                self.fig.clear()
                ax = self.fig.add_subplot(111)

                all_event_activities = []
                all_zscores = []

                for i, (_, event) in enumerate(events.iterrows()):
                    if self.event_time_absolute:
                        start_time = event['start_absolute'] - self.video_start_fiber
                        end_time = event['end_absolute'] - self.video_start_fiber
                    else:
                        start_time = event['start_time']
                        end_time = event['end_time']

                    event_start = start_time - pre_window
                    event_end = end_time + post_window

                    all_event_activities = []

                    for channel_num in self.active_channels:
                        if channel_num in self.zscore_data:
                            zscore_data = self.zscore_data[channel_num]
                            time_col = self.channels.get('time', self.preprocessed_data.columns[0])
                            time_data = self.preprocessed_data[time_col] - self.video_start_fiber

                            event_mask = (time_data >= event_start) & (time_data <= event_end)
                            event_time_relative = time_data[event_mask] - (start_time)
                            event_zscore = zscore_data[event_mask]

                            if len(event_time_relative) > 0:
                                all_event_activities.append(event_zscore)
                                all_zscores.extend(event_zscore)

                    data_range = max(all_zscores) - min(all_zscores) if all_zscores else 5
                    y_spacing = data_range * 0.6
                    y_offset = i * y_spacing
                    
                    if len(all_event_activities) > 0:
                        if len(self.active_channels) > 1:
                            all_event_activities = np.array(all_event_activities).T
                            mean_activity = np.mean(all_event_activities, axis=1)
                            std_activity = np.std(all_event_activities, axis=1)
                            shifted_mean = mean_activity + y_offset
                            shifted_lower = shifted_mean - std_activity
                            shifted_upper = shifted_mean + std_activity

                            ax.plot(event_time_relative, shifted_mean, label=f'Event {i+1}')
                            ax.fill_between(event_time_relative, shifted_lower, shifted_upper, alpha=0.2)
                        else:
                            shifted_zscores = all_event_activities[0] + y_offset
                            ax.plot(event_time_relative, shifted_zscores, label=f'Event {i+1}')

                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                if not events.empty:
                    event_duration = events.iloc[0]['end_time'] - events.iloc[0]['start_time']
                    ax.axvline(event_duration, color='k', linestyle='--', linewidth=1)
                    ax.axvspan(0, event_duration, color='yellow', alpha=0.2)

                    type2_events = self.event_data[self.event_data['Event Type'] == 2]
                    for _, event2 in type2_events.iterrows():
                        if self.event_time_absolute:
                            start2 = event2['start_absolute'] - self.video_start_fiber
                            end2 = event2['end_absolute'] - self.video_start_fiber
                        else:
                            start2 = event2['start_time']
                            end2 = event2['end_time']

                        for _, event1 in events.iterrows():
                            if self.event_time_absolute:
                                event1_start = event1['start_absolute'] - self.video_start_fiber
                                event1_end = event1['end_absolute'] - self.video_start_fiber
                            else:
                                event1_start = event1['start_time']
                                event1_end = event1['end_time']

                            if start2 >= event1_start and end2 <= event1_end:
                                rel_start = start2 - event1_start
                                ax.axvline(rel_start, color='r', linestyle='--', linewidth=1)

                ax.set_title(f"Event-Related Activity (Type: {selected_type})")
                ax.set_xlabel("Time Relative to Event Start (s)")
                ax.set_ylabel("Z-Score")
                ax.grid(False)

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                self.canvas.draw()
                self.set_status(f"Event-related activity plotted for type {selected_type} ({len(selected_indices)} events)")
        except Exception as e:
            messagebox.showerror("Error", f"Event analysis failed: {str(e)}")
            self.set_status("Event analysis failed")

    def plot_experiment_related_activity(self):
        if self.event_data is None and (not self.multi_animal_data or not any('event_data' in a for a in self.multi_animal_data)):
            messagebox.showwarning("No Event Data", "Please load event data first")
            return

        try:
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            selected_type = int(self.event_type.get())
            
            if self.multi_animal_data:
                plot_window = tk.Toplevel(self.root)
                plot_window.title("Experiment-Related Activity (Multi-Animal)")
                plot_window.geometry("1000x800")
                
                tab_control = ttk.Notebook(plot_window)
                groups = {}
                
                for animal_data in self.multi_animal_data:
                    group = animal_data.get('group', 'Unknown')
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(animal_data)
                
                for group_name, animal_list in groups.items():
                    group_event_list = []
                    valid_animal_count = 0
                    
                    for animal_data in animal_list:
                        if 'event_data' not in animal_data or 'zscore_data' not in animal_data:
                            continue
                        
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type]
                        if not events.empty:
                            group_event_list.append((animal_data, events))
                            valid_animal_count += 1
                    
                    if not group_event_list:
                        continue
                    
                    _, ref_events = group_event_list[0]
                    selected_indices = self.select_event_occurrences_dialog(ref_events, group_name)
                    
                    if not selected_indices:
                        continue
                    
                    tab = ttk.Frame(tab_control)
                    tab_control.add(tab, text=group_name)
                    
                    fig = Figure(figsize=(10, 6), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    all_responses = []
                    
                    for animal_data, events in group_event_list:
                        filtered_events = events.iloc[selected_indices]
                        if filtered_events.empty:
                            continue
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        for channel_num in animal_data['active_channels']:
                            if channel_num not in animal_data['zscore_data']:
                                continue
                            
                            zscore_data = animal_data['zscore_data'][channel_num]
                            
                            for _, event in filtered_events.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                    end_time = event['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    start_time = event['start_time']
                                    end_time = event['end_time']
                                
                                duration = end_time - start_time
                                event_start = start_time - pre_window
                                event_end = end_time + post_window
                                
                                event_mask = (time_data >= event_start) & (time_data <= event_end)
                                event_time_relative = time_data[event_mask] - start_time
                                event_zscore = zscore_data[event_mask]
                                
                                if len(event_time_relative) > 0:
                                    all_responses.append((event_time_relative, event_zscore))
                    
                    if not all_responses:
                        continue
                    
                    fps = 30
                    common_time = np.linspace(-pre_window, duration + post_window, 
                                            int((duration + pre_window + post_window) * fps))
                    all_interp = []
                    
                    for time_rel, zscore in all_responses:
                        if len(time_rel) > 1:
                            interp_zscore = np.interp(common_time, time_rel, zscore, 
                                                    left=np.nan, right=np.nan)
                            all_interp.append(interp_zscore)
                    
                    if not all_interp:
                        continue
                    
                    n = np.sum(~np.isnan(all_interp), axis=0)
                    mean_response = np.where(n > 0, np.nanmean(all_interp, axis=0), np.nan)
                    std_dev = np.where(n >= 2, np.nanstd(all_interp, axis=0, ddof=1), np.nan)
                    sem_response = np.where(n >= 2, std_dev / np.sqrt(n), np.nan)
                    
                    ax.plot(common_time, mean_response, 'b-', label='Mean Response')
                    ax.fill_between(common_time, mean_response - sem_response, 
                                    mean_response + sem_response, color='b', alpha=0.2, label='± SEM')
                    
                    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Event Start')
                    ax.axvline(duration, color='k', linestyle='--', linewidth=1, label='Event End')
                    ax.axvspan(0, duration, color='yellow', alpha=0.2, label='Event Duration')
                    
                    for animal_data in animal_list:
                        if 'event_data' not in animal_data:
                            continue
                        
                        events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                        events1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type]
                        events1 = events1.iloc[selected_indices]
                        
                        for _, event2 in events2.iterrows():
                            if animal_data.get('event_time_absolute', False):
                                start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start2 = event2['start_time']
                                end2 = event2['end_time']
                            
                            for _, event1 in events1.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                    event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    event1_start = event1['start_time']
                                    event1_end = event1['end_time']
                                
                                if start2 >= event1_start and end2 <= event1_end:
                                    rel_start = start2 - event1_start
                                    ax.axvline(rel_start, color='r', linestyle='--', linewidth=1,
                                            label='Type 2 Event' if 'Type 2 Event' not in ax.get_legend_handles_labels()[1] else "")
                    
                    ax.set_xlabel("Time Relative to Event Start (s)")
                    ax.set_ylabel("Z-Score")
                    ax.set_title(f"Group: {group_name}\nExperiment-Related Activity "
                                f"({len(all_interp)} events from {valid_animal_count} animals)")
                    ax.grid(False)
                    ax.legend()
                    
                    canvas.draw()
                
                tab_control.pack(expand=1, fill="both")
                self.set_status("Experiment-related activity plotted in new window")
            else:
                if self.zscore_data is None:
                    self.calculate_and_plot_zscore()
                
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                all_responses = []
                
                events = self.event_data[self.event_data['Event Type'] == selected_type]
                selected_indices = self.select_event_occurrences_dialog(events, "Single Animal")
                
                if not selected_indices:
                    messagebox.showinfo("No Selection", "No events selected for analysis")
                    return
                    
                events = events.iloc[selected_indices]
                
                for channel_num in self.active_channels:
                    if channel_num not in self.zscore_data:
                        continue
                    
                    zscore_data = self.zscore_data[channel_num]
                    
                    for _, event in events.iterrows():
                        if self.event_time_absolute:
                            start_time = event['start_absolute'] - self.video_start_fiber
                            end_time = event['end_absolute'] - self.video_start_fiber
                        else:
                            start_time = event['start_time']
                            end_time = event['end_time']

                        duration = end_time - start_time
                        event_start = start_time - pre_window
                        event_end = end_time + post_window
                        
                        event_mask = (time_data >= event_start) & (time_data <= event_end)
                        event_time_relative = time_data[event_mask] - start_time
                        event_zscore = zscore_data[event_mask]
                        
                        if len(event_time_relative) > 0:
                            all_responses.append((event_time_relative, event_zscore))
                
                if not all_responses:
                    return
                
                fps = 30
                common_time = np.linspace(-pre_window, duration + post_window,
                                        int((duration + pre_window + post_window) * fps))
                all_interp = []
                
                for time_rel, zscore in all_responses:
                    if len(time_rel) > 1:
                        interp_zscore = np.interp(common_time, time_rel, zscore,
                                                left=np.nan, right=np.nan)
                        all_interp.append(interp_zscore)
                
                if not all_interp:
                    return
                
                n = np.sum(~np.isnan(all_interp), axis=0)
                mean_response = np.where(n > 0, np.nanmean(all_interp, axis=0), np.nan)
                std_dev = np.where(n >= 2, np.nanstd(all_interp, axis=0, ddof=1), np.nan)
                sem_response = np.where(n >= 2, std_dev / np.sqrt(n), np.nan)
                
                ax.plot(common_time, mean_response, 'b-', label='Mean Response')
                ax.fill_between(common_time, mean_response - sem_response,
                                mean_response + sem_response, color='b', alpha=0.2, label='± SEM')
                
                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                ax.axvline(duration, color='k', linestyle='--', linewidth=1)
                ax.axvspan(0, duration, color='yellow', alpha=0.2)
                
                type2_events = self.event_data[self.event_data['Event Type'] == 2]
                for _, event2 in type2_events.iterrows():
                    if self.event_time_absolute:
                        start2 = event2['start_absolute'] - self.video_start_fiber
                        end2 = event2['end_absolute'] - self.video_start_fiber
                    else:
                        start2 = event2['start_time']
                        end2 = event2['end_time']

                    for _, event1 in events.iterrows():
                        if self.event_time_absolute:
                            event1_start = event1['start_absolute'] - self.video_start_fiber
                            event1_end = event1['end_absolute'] - self.video_start_fiber
                        else:
                            event1_start = event1['start_time']
                            event1_end = event1['end_time']

                        if start2 >= event1_start and end2 <= event1_end:
                            rel_start = start2 - event1_start
                            ax.axvline(rel_start, color='r', linestyle='--', linewidth=1)

                ax.set_xlabel("Time Relative to Event Start (s)")
                ax.set_ylabel("Z-Score")
                ax.set_title(f"Experiment-Related Activity ({len(all_interp)} events)")
                ax.grid(False)
                ax.legend()
                
                self.canvas.draw()
                self.set_status(f"Experiment-related activity plotted for {len(all_interp)} events")
        except Exception as e:
            messagebox.showerror("Error", f"Experiment-related analysis failed: {str(e)}")
            self.set_status("Experiment analysis failed")

    def plot_heatmap(self):
        if self.event_data is None and (not self.multi_animal_data or not any('event_data' in a for a in self.multi_animal_data)):
            messagebox.showwarning("No Event Data", "Please load event data first")
            return
            
        if self.multi_animal_data and not all('zscore_data' in a for a in self.multi_animal_data):
            self.calculate_and_plot_zscore()

        if not self.multi_animal_data and self.zscore_data is None:
            self.calculate_and_plot_zscore()
            
        try:
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            selected_type = int(self.event_type.get())
            
            if pre_window is None or post_window is None or selected_type is None:
                return
            
            if self.multi_animal_data:
                # Group animals by group
                groups = {}
                for animal_data in self.multi_animal_data:
                    group = animal_data.get('group', 'Unknown')
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(animal_data)
                
                # Create a new window for multi-animal plots
                plot_window = tk.Toplevel(self.root)
                plot_window.title("Multi-Animal Activity Heatmap")
                plot_window.geometry("1000x800")
                
                # Create tabs for each group
                tab_control = ttk.Notebook(plot_window)
                
                for group_name, animal_list in groups.items():
                    tab = ttk.Frame(tab_control)
                    tab_control.add(tab, text=group_name)
                    
                    fig = Figure(figsize=(10, 6), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    
                    all_activities = []
                    row_labels = []
                    
                    for animal_idx, animal_data in enumerate(animal_list):
                        if 'event_data' not in animal_data or 'zscore_data' not in animal_data:
                            continue
                        
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type]
                        if events.empty:
                            continue
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        for channel_num in animal_data['active_channels']:
                            if channel_num in animal_data['zscore_data']:
                                zscore_data = animal_data['zscore_data'][channel_num]
                                
                                for event_idx, (_, event) in enumerate(events.iterrows()):
                                    if animal_data.get('event_time_absolute', False):
                                        start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                        end_time = event['end_absolute'] - animal_data['video_start_fiber']
                                    else:
                                        start_time = event['start_time']
                                        end_time = event['end_time']
                                    
                                    event_start = start_time - pre_window
                                    event_end = end_time + post_window
                                    
                                    event_mask = (time_data >= event_start) & (time_data <= event_end)
                                    event_time_relative = time_data[event_mask] - start_time
                                    event_zscore = zscore_data[event_mask]
                                    
                                    interp_time = np.linspace(-pre_window, event_end - start_time, 100)
                                    if len(event_time_relative) > 1:
                                        animal_id = animal_data['animal_id']
                                        interp_zscore = np.interp(interp_time, event_time_relative, event_zscore, 
                                                                left=np.nan, right=np.nan)
                                        all_activities.append(interp_zscore)
                                        row_labels.append(f"{animal_id} CH{channel_num} Event {event_idx+1}")
                    
                    if not all_activities:
                        continue
                        
                    activity_matrix = np.array(all_activities)
                    
                    # Create custom colormap
                    colors = ["blue", "white", "red"]
                    cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors, N=256)
                    
                    # Plot heatmap
                    im = ax.imshow(activity_matrix, aspect='auto', cmap=cmap, 
                                extent=[-pre_window, event_end - start_time, 0, len(all_activities)], 
                                vmin=-3, vmax=3)
                    
                    ax.axvline(0, color='k', linestyle='--', linewidth=1)

                    if not events.empty:
                        event_duration = events.iloc[0]['end_time'] - events.iloc[0]['start_time']
                        ax.axvline(event_duration, color='k', linestyle='--', linewidth=1)
                        # ax.axvspan(0, event_duration, color='yellow', alpha=0.2)

                        for animal_data in animal_list:
                            if 'event_data' in animal_data:
                                events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                                events1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 1]
                                for _, event2 in events2.iterrows():
                                    if animal_data.get('event_time_absolute', False):
                                        start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                        end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                                    else:
                                        start2 = event2['start_time']
                                        end2 = event2['end_time']

                                    for _, event1 in events1.iterrows():
                                        if animal_data.get('event_time_absolute', False):
                                            event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                            event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                        else:
                                            event1_start = event1['start_time']
                                            event1_end = event1['end_time']

                                        if start2 >= event1_start and end2 <= event1_end:
                                            rel_start = start2 - event1_start
                                            ax.axvline(rel_start, color='k', linestyle='--', linewidth=1)
                    
                    ax.set_yticks(np.arange(len(all_activities)) + 0.5)
                    ax.set_yticklabels(row_labels)
                    ax.set_xlabel("Time Relative to Event Start (s)")
                    ax.set_ylabel("Animal/Channel/Event")
                    ax.set_title(f"Group: {group_name} - Event-Related Activity Heatmap")
                    
                    fig.colorbar(im, ax=ax, label="Z-Score")
                    fig.tight_layout()
                    canvas.draw()
                
                tab_control.pack(expand=1, fill="both")
                self.set_status("Multi-animal heatmap plotted in new window")
            else:
                # Single animal mode (existing code)
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
                    if channel_num not in self.zscore_data:
                        continue
                        
                    ax = self.fig.add_subplot(num_channels, 1, idx+1)
                    
                    zscore_data = self.zscore_data[channel_num]
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
                        event_time_relative = time_data[event_mask] - start_time
                        event_zscore = zscore_data[event_mask]
                        
                        interp_time = np.linspace(-pre_window, event_end - start_time, 100)
                        interp_zscore = np.interp(interp_time, event_time_relative, event_zscore)
                        
                        all_activities.append(interp_zscore)
                    
                    if not all_activities:
                        continue
                        
                    activity_matrix = np.array(all_activities)
                    
                    colors = ["blue", "white", "red"]
                    cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors, N=256)
                    
                    event_duration = events.iloc[0]['end_time'] - events.iloc[0]['start_time']
                    im = ax.imshow(activity_matrix, aspect='auto', cmap=cmap, 
                                extent=[-pre_window, event_duration + post_window, 
                                        0, len(events)], vmin=-3, vmax=3)
                    
                    ax.axvline(0, color='k', linestyle='--', linewidth=1)
                    ax.axvline(event_duration, color='k', linestyle='--', linewidth=1)
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
                    
                    ax.set_title(f"Channel {channel_num}")
                    if idx == num_channels - 1:
                        ax.set_xlabel("Time Relative to Event Start (s)")
                    ax.set_ylabel("Event #")
                    ax.grid(False)
                    
                    self.fig.colorbar(im, ax=ax, label="Z-Score")
                
                self.fig.suptitle(f"Event-Related Activity Heatmap (Type: {selected_type})")
                self.fig.tight_layout(rect=[0, 0, 1, 0.95])
                self.canvas.draw()
                self.set_status(f"Heatmap plotted for type {selected_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Heatmap creation failed: {str(e)}")
            self.set_status("Heatmap failed")
    
    def export_statistic_results(self):
        """Export statistics (peak and AUC) for event-related activity"""
        if self.event_data is None and (not self.multi_animal_data or not any('event_data' in a for a in self.multi_animal_data)):
            messagebox.showwarning("No Event Data", "Please load event data first")
            return
            
        if self.multi_animal_data and not all('zscore_data' in a for a in self.multi_animal_data):
            self.calculate_and_plot_zscore()
            
        if not self.multi_animal_data and self.zscore_data is None:
            self.calculate_and_plot_zscore()
            
        # Create dialog for parameter input
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Statistics Parameters")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Pre-event window (s):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        pre_entry = ttk.Entry(main_frame, width=10)
        pre_entry.insert(0, "2")
        pre_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Post-event window (s):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        post_entry = ttk.Entry(main_frame, width=10)
        post_entry.insert(0, "5")
        post_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Event Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        event_type_entry = ttk.Entry(main_frame, width=10)
        event_type_entry.insert(0, "1")
        event_type_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Signal Type:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        signal_type_var = tk.StringVar(value="zscore")
        ttk.Radiobutton(main_frame, text="Z-score", variable=signal_type_var, value="zscore").grid(row=3, column=1, sticky="w", padx=5)
        ttk.Radiobutton(main_frame, text="ΔF/F", variable=signal_type_var, value="dff").grid(row=4, column=1, sticky="w", padx=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        def on_confirm():
            try:
                pre_window = float(pre_entry.get())
                post_window = float(post_entry.get())
                selected_type = int(event_type_entry.get())
                signal_type = signal_type_var.get()
                
                dialog.destroy()
                self.calculate_and_export_stats(pre_window, post_window, selected_type, signal_type)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for windows and event type")
        
        ttk.Button(btn_frame, text="Confirm", command=on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
    def calculate_and_export_stats(self, pre_window, post_window, event_type, signal_type):
        """Calculate statistics and export to CSV"""
        try:
            all_stats = []
            
            if self.multi_animal_data:
                # Multi-animal mode
                for animal_data in self.multi_animal_data:
                    if 'event_data' not in animal_data or 'zscore_data' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    group = animal_data.get('group', 'Unknown')
                    
                    events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                    if events.empty:
                        continue
                    
                    selected_indices = self.select_event_occurrences_dialog(events, f"{animal_id} ({group})")
                    if not selected_indices:
                        continue
                    
                    filtered_events = events.iloc[selected_indices]
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num not in animal_data['zscore_data']:
                            continue
                            
                        # Get signal data based on selected type
                        if signal_type == "zscore":
                            signal_data = animal_data['zscore_data'][channel_num]
                        else:  # dff
                            signal_data = animal_data['dff_data'][channel_num]
                        
                        for event_idx, (_, event) in enumerate(filtered_events.iterrows()):
                            if animal_data.get('event_time_absolute', False):
                                start_time = event['start_absolute'] - animal_data['video_start_fiber']
                            else:
                                start_time = event['start_time']
                            
                            event_start = start_time - pre_window
                            event_end = start_time + post_window
                            
                            event_mask = (time_data >= event_start) & (time_data <= event_end)
                            event_time_relative = time_data[event_mask] - start_time
                            event_signal = signal_data[event_mask]
                            
                            if len(event_signal) > 0:
                                # Calculate peak value
                                peak_value = np.max(event_signal)
                                
                                # Calculate AUC using trapezoidal integration
                                auc_value = np.trapezoid(event_signal, event_time_relative)
                                
                                all_stats.append({
                                    'Group': group,
                                    'Animal ID': animal_id,
                                    'Channel': channel_num,
                                    'Event Index': event_idx + 1,
                                    'Event Start': start_time,
                                    'Peak Value': peak_value,
                                    'AUC': auc_value,
                                    'Signal Type': signal_type
                                })
            else:
                # Single animal mode
                events = self.event_data[self.event_data['Event Type'] == event_type]
                if events.empty:
                    messagebox.showwarning("No Events", f"No events of type {event_type} found")
                    return
                    
                selected_indices = self.select_event_occurrences_dialog(events, "Single Animal")
                if not selected_indices:
                    return
                    
                events = events.iloc[selected_indices]
                
                time_col = self.channels['time']
                time_data = self.preprocessed_data[time_col] - self.video_start_fiber
                
                for channel_num in self.active_channels:
                    if channel_num not in self.zscore_data:
                        continue
                        
                    # Get signal data based on selected type
                    if signal_type == "zscore":
                        signal_data = self.zscore_data[channel_num]
                    else:  # dff
                        signal_data = self.dff_data[channel_num]
                    
                    for event_idx, (_, event) in enumerate(events.iterrows()):
                        if self.event_time_absolute:
                            start_time = event['start_absolute'] - self.video_start_fiber
                        else:
                            start_time = event['start_time']
                        
                        event_start = start_time - pre_window
                        event_end = start_time + post_window
                        
                        event_mask = (time_data >= event_start) & (time_data <= event_end)
                        event_time_relative = time_data[event_mask] - start_time
                        event_signal = signal_data[event_mask]
                        
                        if len(event_signal) > 0:
                            # Calculate peak value
                            peak_value = np.max(event_signal)
                            
                            # Calculate AUC using trapezoidal integration
                            auc_value = np.trapezoid(event_signal, event_time_relative)
                            
                            all_stats.append({
                                'Group': 'Single',
                                'Animal ID': 'Single Animal',
                                'Channel': channel_num,
                                'Event Index': event_idx + 1,
                                'Event Start': start_time,
                                'Peak Value': peak_value,
                                'AUC': auc_value,
                                'Signal Type': signal_type
                            })
            
            if not all_stats:
                messagebox.showinfo("No Data", "No valid statistics calculated")
                return
                
            # Create DataFrame and save to CSV
            df = pd.DataFrame(all_stats)
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Statistics Results"
            )
            
            if save_path:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Statistics exported to:\n{save_path}")
                self.set_status(f"Statistics exported: {len(all_stats)} records")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate statistics: {str(e)}")
            self.set_status("Statistics export failed")
    
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
        self.multi_animal_mode = tk.BooleanVar(value=False)
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Freezing Analyzer")
        self.eztrack_results_path = ""
        self.freezing_data = {}
        self.window_freeze = None
        self.selected_files = []
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)

        mode_frame = ttk.Frame(control_panel)
        mode_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Analysis Mode:").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Single", variable=self.multi_animal_mode, value=False,
                       command=self.toggle_analysis_mode).pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Multi", variable=self.multi_animal_mode, value=True,
                       command=self.toggle_analysis_mode).pack(side="left", padx=5)
        
        self.multi_animal_frame = ttk.LabelFrame(control_panel, text="Multi Animal Analysis")
        
        list_frame = ttk.Frame(self.multi_animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(self.multi_animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Add Selected", command=self.add_single_file).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Import All", command=self.import_multi_animal_data).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Plot Raster", command=self.plot_multi_animal_raster).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        
        self.single_animal_frame = ttk.LabelFrame(control_panel, text="Freezing Analysis")
        button_frame = ttk.Frame(self.single_animal_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="One-click Import", command=self.one_click_import, style="Accent.TButton").grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Plot Freezing Timeline", command=self.plot_freezing_windows).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Export Freezing", command=self.export_freezing_csv).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            fiber_frame.columnconfigure(0, weight=1)
            fiber_frame.columnconfigure(1, weight=1)
            
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            ttk.Button(fiber_frame, text="Plot Freezing w/Fiber", command=self.plot_freezing_with_fiber).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        self.toggle_analysis_mode()
    
    def toggle_analysis_mode(self):
        if self.multi_animal_mode.get():
            self.multi_animal_frame.pack(fill="x", padx=5, pady=5)
            self.single_animal_frame.pack_forget()
        else:
            self.multi_animal_frame.pack_forget()
            self.single_animal_frame.pack(fill="x", padx=5, pady=5)

    def add_single_file(self):
        folder_path = filedialog.askdirectory(title="Select animal data folder")
        if not folder_path:
            return

        try:
            folder_path = os.path.normpath(folder_path)
            path_parts = folder_path.split(os.sep)
            if len(path_parts) < 4:
                messagebox.showwarning("Invalid Directory", "Selected directory is not a valid animal data folder")
                return

            batch_name = path_parts[-3]
            group = path_parts[-2]
            ear_tag = path_parts[-1]
            animal_id = f"{batch_name}-{ear_tag}"

            events_file = None
            eztrack_file = None
            fiber_file = None

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if not os.path.isfile(file_path):
                    continue

                file_lower = file_name.lower()
                if fnmatch.fnmatch(file_lower, "*events*.csv") or fnmatch.fnmatch(file_lower, "*timeline*.csv"):
                    events_file = file_path
                elif fnmatch.fnmatch(file_lower, "*freezingoutput.csv") or fnmatch.fnmatch(file_lower, "*eztrack*.csv"):
                    eztrack_file = file_path
                elif fnmatch.fnmatch(file_lower, "fluorescence.csv") or fnmatch.fnmatch(file_lower, "*fiber*.csv"):
                    fiber_file = file_path

            animal_data = {
                'animal_id': animal_id,
                'group': group,
                'events_path': events_file,
                'eztrack_path': eztrack_file,
                'fiber_path': fiber_file
            }
            
            if events_file and eztrack_file:
                self.selected_files.append(animal_data)
                self.file_listbox.insert(tk.END, f"{animal_id} ({group})")
                self.process_animal_data(animal_data)
            else:
                messagebox.showwarning("No Data", "No valid events or eztrack files found in the selected directory")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add single file: {str(e)}")
    
    def clear_selected(self):
        selected_indices = self.file_listbox.curselection()
        for index in sorted(selected_indices, reverse=True):
            animal_id = self.file_listbox.get(index)
            self.file_listbox.delete(index)
            
            # Remove from selected_files and multi_animal_data
            if index < len(self.selected_files):
                animal_data = self.selected_files.pop(index)
                # Find and remove from multi_animal_data
                for i, data in enumerate(self.multi_animal_data):
                    if data['animal_id'] == animal_data['animal_id']:
                        self.multi_animal_data.pop(i)
                        break
    
    def clear_all(self):
        self.file_listbox.delete(0, tk.END)
        self.selected_files = []
        self.multi_animal_data = []
    
    def analyze_multi_animal(self):
        if not self.selected_files:
            messagebox.showwarning("No Data", "Please select animals first")
            return
        
        self.set_status("Analyzing multi-animal data...")
        self.multi_animal_results = {}
        
        for animal_data in self.selected_files:
            animal_id = animal_data['animal_id']
            group = animal_data['group']
            
            if group not in self.multi_animal_results:
                self.multi_animal_results[group] = []
            
            try:
                event_data = pd.read_csv(animal_data['events_path'])
                start_offset = event_data['start_time'].iloc[0]
                event_data['start_time'] = event_data['start_time'] - start_offset
                event_data['end_time'] = event_data['end_time'] - start_offset
                
                eztrack_result = pd.read_csv(animal_data['eztrack_path'])
                
                freezing_percentages = {}
                for idx, event in event_data.iterrows():
                    event_start = event['start_time'] * 30  # FPS=30
                    event_end = event['end_time'] * 30
                    
                    event_frames = eztrack_result[
                        (eztrack_result['Frame'] >= event_start) & 
                        (eztrack_result['Frame'] <= event_end)]
                    
                    if len(event_frames) > 0:
                        freezing_frames = event_frames[event_frames['Freezing'] == 100]
                        freezing_percentage = len(freezing_frames) / len(event_frames) * 100
                    else:
                        freezing_percentage = 0
                    
                    freezing_percentages[f"Event_{idx+1}"] = round(freezing_percentage, 2)
                
                window_size_frames = 30 * 30
                total_frames = len(eztrack_result)
                num_windows = math.ceil(total_frames / window_size_frames)
                
                freeze_sec = []
                window_times = []
                
                for i in range(num_windows):
                    start_frame = i * window_size_frames
                    end_frame = (i + 1) * window_size_frames - 1
                    
                    window_frames = eztrack_result[
                        (eztrack_result['Frame'] >= start_frame) & 
                        (eztrack_result['Frame'] <= end_frame)]
                    
                    if len(window_frames) > 0:
                        freezing_frames = window_frames[window_frames['Freezing'] == 100]
                        freezing_seconds = len(freezing_frames) / 30
                    else:
                        freezing_seconds = 0
                    
                    freeze_sec.append(freezing_seconds)
                    window_center = (start_frame + end_frame) / (2 * 30)
                    window_times.append(window_center)
                
                self.multi_animal_results[group].append({
                    'animal_id': animal_id,
                    'freezing_percentages': freezing_percentages,
                    'window_times': window_times,
                    'freeze_sec': freeze_sec,
                    'event_data': event_data,
                    'eztrack_result': eztrack_result
                })
            except Exception as e:
                self.set_status(f"Error analyzing {animal_id}: {str(e)}")
        
        self.set_status("Multi-animal analysis completed")
        return self.multi_animal_results
    
    def plot_multi_animal_raster(self):
        if not self.selected_files:
            messagebox.showwarning("No Data", "Please select animals first")
            return
        
        results = self.analyze_multi_animal()
        if not results:
            return
        
        for group in ["CFC", "EXT", "RET", "CFT"]:
            if group not in results or not results[group]:
                continue
            
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            y_ticks = []
            y_labels = []
            y_pos = 0
            
            for animal_data in results[group]:
                animal_id = animal_data['animal_id']
                eztrack_result = animal_data['eztrack_result']
                event_data = animal_data['event_data']
                
                y_ticks.append(y_pos)
                y_labels.append(animal_id)
                
                time_values = eztrack_result['Frame'] / 30
                freezing_values = eztrack_result['Freezing']
                
                freezing_times = time_values[freezing_values == 100]
                
                for _, event in event_data.iterrows():
                    event_type = event['Event Type']
                    color = "#ffffff" if event_type == 0 else "#0400fc"

                    ax.add_patch(plt.Rectangle(
                        (event['start_time'], y_pos - 0.4),
                        event['end_time'] - event['start_time'],
                        0.8, color=color, alpha=0.5
                    ))

                for t in freezing_times:
                    ax.vlines(t, y_pos - 0.4, y_pos + 0.4, color='black', alpha=0.5, linewidth=0.5)

                y_pos += 1
            
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Animals")
            ax.set_title(f"Freezing Raster Plot - {group}")
            ax.grid(False)
            
            top = tk.Toplevel()
            top.title(f"Freezing Raster Plot - {group}")
            
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            toolbar = NavigationToolbar2Tk(canvas, top)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
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
        
        smooth_frame = ttk.Frame(self.step2_frame)
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(smooth_frame, text="Apply Smoothing", variable=self.apply_smooth,
                       command=lambda: self.toggle_widgets(smooth_frame, self.apply_smooth.get(), 1)).grid(row=0, column=0, sticky="w")
        
        param_frame = ttk.Frame(smooth_frame)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Label(param_frame, text="Window Size:").grid(row=0, column=0, sticky="w")
        ttk.Scale(param_frame, from_=3, to=101, orient=tk.HORIZONTAL, 
                 length=100, variable=self.smooth_window,
                 command=lambda v: self.smooth_window.set(int(float(v)))).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_window).grid(row=0, column=2)
        
        ttk.Label(param_frame, text="Polynomial Order:").grid(row=1, column=0, sticky="w")
        ttk.Scale(param_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                 length=100, variable=self.smooth_order,
                 command=lambda v: self.smooth_order.set(int(float(v)))).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_order).grid(row=1, column=2)
        
        param_frame.grid_remove()
        
        baseline_corr_frame = ttk.Frame(self.step2_frame)
        baseline_corr_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=self.apply_baseline,
                        command=lambda: self.toggle_widgets(baseline_corr_frame, self.apply_baseline.get(), 1)).grid(row=0, column=0, sticky="w")
        
        model_frame = ttk.Frame(baseline_corr_frame)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        ttk.Label(model_frame, text="Baseline Model:").grid(row=1, column=0, sticky="w")
        model_options = ["Polynomial", "Exponential"]
        model_menu = ttk.OptionMenu(model_frame, self.baseline_model, "Polynomial", *model_options)
        model_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        model_frame.grid_remove()
        
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
        try:
            if self.multi_animal_data:
                if self.apply_smooth.get():
                    self.smooth_data()
                if self.apply_baseline.get():
                    self.baseline_correction()
                if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                    self.motion_correction()
                
                messagebox.showinfo("Success", "Preprocessing applied to all animals")
                self.set_status("Preprocessing applied to all animals")
                self.plot_preprocessed()
            
            else:
                if self.fiber_cropped is None:
                    messagebox.showwarning("No Data", "Please load fiber data first")
                    return
                
                self.preprocessed_data = self.fiber_cropped.copy()
                
                if self.apply_smooth.get():
                    self.smooth_data()
                if self.apply_baseline.get():
                    self.baseline_correction()
                if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                    self.motion_correction()
                
                messagebox.showinfo("Success", "Preprocessing applied successfully")
                self.set_status("Preprocessing applied")
                self.plot_preprocessed()
                
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.set_status("Preprocessing failed")
    
    def show_step3(self):
        if self.multi_animal_data:
            # Always show for multi-animal
            pass
        elif self.preprocessed_data is None:
            messagebox.showwarning("No Data", "Please preprocess data first")
            return
            
        self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        self.step3_frame.columnconfigure(0, weight=1)
        self.step3_frame.columnconfigure(1, weight=1)
        
        ttk.Button(self.step3_frame, text="Calculate & plot ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate & plot Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
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
        self.event_type.insert(0, "1")
        self.event_type.grid(row=2, column=1, padx=5)
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text="Plot Activity1", 
                  command=self.plot_event_related_activity).grid(row=0, column=0, sticky="ew")
        
        ttk.Button(btn_frame, text="Plot Activity2", 
                  command=self.plot_experiment_related_activity).grid(row=0, column=1, sticky="ew")
        
        ttk.Button(btn_frame, text="Plot Heatmap", 
                  command=self.plot_heatmap).grid(row=1, column=0, sticky="ew")
        
        ttk.Button(btn_frame, text="Export Statistic", 
                  command=self.export_statistic_results).grid(row=1, column=1, sticky="ew")
    
    def plot_preprocessed(self):
        """Plot preprocessed data"""
        if self.multi_animal_data:
            # Plot all animals
            print("No need to figure it out!")  # Reuse raw data plot which now handles multi-animal
            return
            
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
        if not self.multi_animal_data and (not hasattr(self, 'window_freeze') or self.window_freeze is None):
            messagebox.showwarning("No Data", "Please run freezing analysis first")
            return
        
        try:
            self.fig.clear()
            
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = self.fig.add_subplot(gs[0])
            ax2 = self.fig.add_subplot(gs[1])
            
            if self.multi_animal_data:
                all_signals = []
                common_time = None
                
                for animal_data in self.multi_animal_data:
                    if 'fiber_cropped' not in animal_data or not animal_data.get('active_channels'):
                        continue
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['fiber_cropped'][time_col] - animal_data['video_start_fiber']
                    
                    channel_num = animal_data['active_channels'][0]
                    target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                    if target_col and target_col in animal_data['fiber_cropped'].columns:
                        signal = animal_data['fiber_cropped'][target_col]
                        
                        if common_time is None:
                            min_time = np.min(time_data)
                            max_time = np.max(time_data)
                            common_time = np.linspace(min_time, max_time, 1000)
                        
                        interp_signal = np.interp(common_time, time_data, signal)
                        all_signals.append(interp_signal)
                
                if all_signals:
                    mean_signal = np.mean(all_signals, axis=0)
                    std_signal = np.std(all_signals, axis=0)
                    
                    ax1.plot(common_time, mean_signal, 'b-', label='Mean Fiber Signal')
                    ax1.fill_between(common_time, mean_signal - std_signal, 
                                    mean_signal + std_signal, color='b', alpha=0.2)
            else:
                time_col = self.channels['time']
                time_data = self.fiber_cropped[time_col] - self.video_start_fiber
                
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                for i, channel_num in enumerate(self.active_channels):
                    if channel_num in self.channel_data:
                        target_col = self.channel_data[channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in self.fiber_cropped.columns:
                            color = colors[i % len(colors)]
                            ax1.plot(time_data, self.fiber_cropped[target_col], 
                                   f'{color}-', linewidth=1, label=f'CH{channel_num} Fiber Data')
            
            ax1.set_title('Freezing Analysis with Fiber Photometry')
            ax1.set_ylabel("Fiber Signal")
            ax1.grid(False)
            ax1.legend()
            
            if self.multi_animal_data:
                y_ticks = []
                y_labels = []
                
                for idx, animal_data in enumerate(self.multi_animal_data):
                    if 'eztrack_result' not in animal_data:
                        continue
                    
                    eztrack_result = animal_data['eztrack_result']
                    time_values = eztrack_result['Frame'] / 30
                    freezing_values = eztrack_result['Freezing']
                    
                    freezing_times = time_values[freezing_values == 100]
                    
                    y_pos = idx
                    y_ticks.append(y_pos)
                    y_labels.append(animal_data['animal_id'])
                    
                    for t in freezing_times:
                        ax2.vlines(t, y_pos - 0.4, y_pos + 0.4, color='black', 
                                  linewidth=0.5, alpha=0.7)
                
                ax2.set_yticks(y_ticks)
                ax2.set_yticklabels(y_labels)
            else:
                time_values = self.eztrack_result['Frame'] / 30
                freezing_values = self.eztrack_result['Freezing']
                freezing_times = time_values[freezing_values == 100]
                
                y_pos = 0
                for t in freezing_times:
                    ax2.vlines(t, y_pos - 0.4, y_pos + 0.4, color='black', 
                              linewidth=0.5, alpha=0.7)
                
                ax2.set_yticks([0])
                ax2.set_yticklabels([self.selected_files[0]['animal_id'] if self.selected_files else 'Animal'])
            
            event_data = self.event_data if not self.multi_animal_data else self.multi_animal_data[0].get('event_data')
            if event_data is not None:
                event_time_absolute = self.event_time_absolute if not self.multi_animal_data else self.multi_animal_data[0].get('event_time_absolute')
                
                if event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                for _, row in event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - (self.video_start_fiber if not self.multi_animal_data else self.multi_animal_data[0]['video_start_fiber'])
                    end_time = row[end_col] - (self.video_start_fiber if not self.multi_animal_data else self.multi_animal_data[0]['video_start_fiber'])
                    
                    ax1.axvspan(start_time, end_time, color=color, alpha=0.3)
                    ax2.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Animals")
            ax2.grid(False)
            
            self.canvas.draw()
            self.set_status("Freezing with fiber plotted as raster")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot freezing with fiber: {str(e)}")
            self.set_status("Plot failed")

class PupilAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Pupil Analyzer")
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

        ttk.Button(pupil_frame, text="Plot Pupil Distance", command=self.plot_pupil_distance).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        ttk.Button(pupil_frame, text="Plot Combined Data", command=self.plot_combined_data).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
            ttk.Button(fiber_frame, text="Plot Pupil w/Fiber", command=self.plot_pupil_with_fiber).grid(row=0, column=1, sticky="ew", padx=5, pady=2)

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
        ttk.Scale(param_frame, from_=1, to=7, orient=tk.HORIZONTAL, 
                 length=150, variable=self.smooth_order,
                 command=lambda v: self.smooth_order.set(int(float(v)))).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_order).grid(row=1, column=2)
        
        param_frame.grid_remove()
        
        baseline_corr_frame = ttk.Frame(self.step2_frame)
        baseline_corr_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=self.apply_baseline,
                        command=lambda: self.toggle_widgets(baseline_corr_frame, self.apply_baseline.get(), 1)).grid(row=0, column=0, sticky="w")
        
        model_frame = ttk.Frame(baseline_corr_frame)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        ttk.Label(model_frame, text="Baseline Model:").grid(row=1, column=0, sticky="w")
        model_options = ["Polynomial", "Exponential"]
        model_menu = ttk.OptionMenu(model_frame, self.baseline_model, "Polynomial", *model_options)
        model_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        model_frame.grid_remove()

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
            if self.apply_smooth.get():
                self.smooth_data()
            
            if self.apply_baseline.get():
                self.baseline_correction()
            
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