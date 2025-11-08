# -*- coding: utf-8 -*-
""" 
Created on Fri Sep 5 20:50:50 2025 

This custom code involved the analysis of freezing and freezing + fiber through the process of deeplabcut.

@author: Pulee 
""" 

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
import signal
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import fnmatch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from scipy.interpolate import interp1d
from itertools import groupby
import traceback
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread")

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
        
        version_label = ttk.Label(main_frame, text="Version 4.0 | © 2025 ML Lab", 
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
        self.baseline_model = tk.StringVar(value="Polynomial")
        self.multi_animal_data = []
        self.current_animal_index = 0
        
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

    def import_multi_animal_data(self):
        base_dir = filedialog.askdirectory(title="Select Free Moving/Behavioural directory")
        if not base_dir:
            return

        exp_mode = "freezing+fiber" if self.include_fiber else "freezing"
        if "PupilAnalyzerApp" in str(type(self)):
            exp_mode = "pupil+fiber" if self.include_fiber else "pupil"

        try:
            self.set_status("Scanning for multi-animal data...")
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
                            'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
                            'events': ['*events*.csv', '*timeline*.csv'],
                            'fiber': ['fluorescence.csv', '*fiber*.csv'],
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

                        required_files = ['dlc', 'events'] if 'freezing' in exp_mode else ['dlc', 'timestamp']
                        if all(ft in files_found for ft in required_files):
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

                            # Process DLC file
                            if 'dlc' in files_found:
                                try:
                                    dlc_data = pd.read_csv(files_found['dlc'], header=[0,1,2])
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

            if not self.multi_animal_data:
                messagebox.showwarning("No Data", "No valid animal data found in the selected directory")
                self.set_status("No multi-animal data found")
            else:
                self.set_status(f"Found {len(self.multi_animal_data)} animals")
                if self.include_fiber:
                    self.show_channel_selection_dialog()
                else:
                    messagebox.showinfo("Success", f"Found and processed {len(self.multi_animal_data)} animals")
                    self.set_status(f"Imported {len(self.multi_animal_data)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            self.set_status("Import failed")

    def add_single_file(self):
        folder_path = filedialog.askdirectory(title="Select animal ear tag folder")
        if not folder_path:
            return

        exp_mode = "freezing+fiber" if self.include_fiber else "freezing"
        if "PupilAnalyzerApp" in str(type(self)):
            exp_mode = "pupil+fiber" if self.include_fiber else "pupil"
        
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

            patterns = {
                'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
                'events': ['*events*.csv', '*timeline*.csv'],
                'fiber': ['fluorescence.csv', '*fiber*.csv'],
                'timestamp': ['*timestamp*.csv', '*time*.csv'],
                'ast2': ['*.ast2']
            }

            required_files = {
                'freezing': ['dlc', 'events'],
                'freezing+fiber': ['dlc', 'events', 'fiber'],
                'pupil': ['dlc', 'timestamp', 'ast2'],
                'pupil+fiber': ['dlc', 'timestamp', 'ast2', 'fiber']
            }[exp_mode]

            files_found = {}
            for file_type, file_patterns in patterns.items():
                found_file = None
                for root_path, dirs, files in os.walk(folder_path):
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

            missing_files = [ft for ft in required_files if ft not in files_found]
            if missing_files:
                messagebox.showwarning("Missing Files", 
                                    f"Required files missing: {', '.join(missing_files)}\nfor mode: {exp_mode}")
                return

            animal_data = {
                'animal_id': animal_id,
                'group': group,
                'files': files_found,
                'processed': False,
                'event_time_absolute': False
            }

            if 'events' in files_found:
                try:
                    event_data = pd.read_csv(files_found['events'])
                    start_offset = event_data['start_time'].iloc[0]
                    event_data['start_time'] = event_data['start_time'] - start_offset
                    event_data['end_time'] = event_data['end_time'] - start_offset
                    animal_data['event_data'] = event_data
                except Exception as e:
                    print(f"Failed to load events for {animal_id}: {str(e)}")

            if 'dlc' in files_found:
                try:
                    dlc_data = pd.read_csv(files_found['dlc'], header=[0,1,2])
                    animal_data['dlc_data'] = dlc_data
                    filename = os.path.basename(files_found['dlc'])
                    match = re.search(r'cam(\d+)', filename)
                    if match:
                        animal_data['cam_id'] = int(match.group(1))
                except Exception as e:
                    print(f"Failed to load DLC for {animal_id}: {str(e)}")

            if 'timestamp' in files_found:
                try:
                    timestamps = pd.read_csv(files_found['timestamp'])
                    exp_start = timestamps[(timestamps['Device'] == 'Experiment') &
                                        (timestamps['Action'] == 'Start')]['Timestamp'].values
                    if len(exp_start) > 0:
                        animal_data['experiment_start'] = exp_start[0]
                except Exception as e:
                    print(f"Failed to load timestamp for {animal_id}: {str(e)}")

            if 'ast2' in files_found:
                try:
                    ast2_data = self.h_AST2_readData(files_found['ast2'])
                    animal_data['ast2_data'] = ast2_data
                except Exception as e:
                    print(f"Failed to load AST2 for {animal_id}: {str(e)}")

            if 'fiber' in files_found:
                try:
                    fiber_result = self.load_fiber_data(files_found['fiber'])
                    if fiber_result:
                        animal_data.update(fiber_result)
                except Exception as e:
                    print(f"Failed to load fiber for {animal_id}: {str(e)}")

            animal_data['processed'] = True
            self.multi_animal_data.append(animal_data)
            self.file_listbox.insert(tk.END, f"{animal_id} ({group})")
            
            if self.include_fiber and 'fiber_data' in animal_data:
                self.show_channel_selection_dialog()
            
            self.set_status(f"Added animal: {animal_id} ({group})")
            messagebox.showinfo("Success", f"Successfully added animal:\n{animal_id} ({group})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add animal: {str(e)}")
            self.set_status("Add animal failed")

    def clear_selected(self):
        selected_indices = self.file_listbox.curselection()
        for index in sorted(selected_indices, reverse=True):
            self.file_listbox.delete(index)
            if index < len(self.multi_animal_data):
                self.multi_animal_data.pop(index)
    
    def select_all(self):
        self.file_listbox.select_set(0, tk.END)

    def load_fiber_data(self, file_path=None):
        path = file_path or self.fiber_data_path
        try:
            fiber_data = pd.read_csv(path, skiprows=1, delimiter=',')
            fiber_data = fiber_data.loc[:, ~fiber_data.columns.str.contains('^Unnamed')]
            fiber_data.columns = fiber_data.columns.str.strip()

            time_col = None
            possible_time_columns = ['timestamp', 'timems', 'time', 'time(ms)']
            for col in fiber_data.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in possible_time_columns):
                    time_col = col
                    break
            
            if not time_col:
                numeric_cols = fiber_data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    time_col = numeric_cols[0]
            
            fiber_data[time_col] = fiber_data[time_col] / 1000

            global fps_fiber
            fps_fiber = int(1 / np.mean(np.diff(fiber_data[time_col])))
            
            events_col = None
            for col in fiber_data.columns:
                if 'event' in col.lower():
                    events_col = col
                    break
            
            channels = {'time': time_col, 'events': events_col}
            
            channel_data = {}
            channel_pattern = re.compile(r'CH(\d+)-(\d+)', re.IGNORECASE)
            
            for col in fiber_data.columns:
                match = channel_pattern.match(col)
                if match:
                    channel_num = int(match.group(1))
                    wavelength = int(match.group(2))
                    
                    if channel_num not in channel_data:
                        channel_data[channel_num] = {'410': None, '415': None, '470': None, '560': None}
                    
                    if wavelength == 410 or wavelength == 415:
                        channel_data[channel_num]['410' if wavelength == 410 else '415'] = col
                    elif wavelength == 470:
                        channel_data[channel_num]['470'] = col
                    elif wavelength == 560:
                        channel_data[channel_num]['560'] = col
            
            self.set_status(f"Fiber data loaded, {len(channel_data)} channels detected")
            
            return {
                'fiber_data': fiber_data,
                'channels': channels,
                'channel_data': channel_data
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fiber data: {str(e)}")
            self.set_status("Fiber data load failed")
            return None
    
    def show_channel_selection_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Channels")
        dialog.geometry("290x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=5)
        main_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(main_frame)
        canvas.config(height=50, width=100)
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
                chk.pack(anchor="w", padx=2, pady=2)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Select All", command=lambda: self.toggle_all_channels(True)).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: self.toggle_all_channels(False)).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Confirm", command=lambda: self.finalize_channel_selection(dialog)).grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        
    def toggle_all_channels(self, select):
        for animal_vars in self.channel_vars.values():
            for var in animal_vars.values():
                var.set(select)

    def finalize_channel_selection(self, dialog):
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

        dialog.destroy()
        self.set_status(f"Selected channels for all animals")
        messagebox.showinfo("Success", f"Processed {len(self.multi_animal_data)} animals")
        self.set_status(f"Imported {len(self.multi_animal_data)} animals")
    
    def set_video_markers(self, animal_data=None):
        try:
            fiber_data = animal_data.get('fiber_data')
            event_data = animal_data.get('event_data')
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            
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
            video_end_fiber = input1_events[time_col].iloc[-2]
            global video_duration
            video_duration = video_end_fiber - video_start_fiber
            
            fiber_cropped = fiber_data[
                (fiber_data[time_col] >= video_start_fiber) & 
                (fiber_data[time_col] <= video_end_fiber)].copy()
            
            animal_data.update({
                'fiber_cropped': fiber_cropped,
                'video_start_fiber': video_start_fiber,
                'video_end_fiber': video_end_fiber
            })
            
            if event_data is not None and not animal_data.get('event_time_absolute'):
                event_data['start_absolute'] = video_start_fiber + event_data['start_time']
                event_data['end_absolute'] = video_start_fiber + event_data['end_time']
                animal_data['event_time_absolute'] = True
            
            self.set_status("Video markers set from fiber data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set video markers: {str(e)}")
            self.set_status("Video markers failed")
    
    def get_selected_animals(self):
        """Get currently selected animals from listbox"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one animal from the list")
            return []
        return [self.multi_animal_data[i] for i in selected_indices]
    
    def plot_raw_data(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        try:
            groups = {}
            for animal_data in selected_animals:
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Raw Data")
            plot_window.geometry("1000x800")
            
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
                
                event_data = animal_list[0].get('event_data')
                if event_data is not None:
                    event_time_absolute = animal_list[0].get('event_time_absolute', False)
                    if event_time_absolute:
                        start_col = 'start_absolute'
                        end_col = 'end_absolute'
                    else:
                        start_col = 'start_time'
                        end_col = 'end_time'
                    
                    video_start = animal_list[0]['video_start_fiber']
                    
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
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            self.set_status("Raw plot failed")
    
    def smooth_data(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        try:
            window_size = self.smooth_window.get()
            poly_order = self.smooth_order.get()
            
            if window_size % 2 == 0:
                window_size += 1
            
            for animal_data in selected_animals:
                animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                for channel_num in animal_data['active_channels']:
                    if channel_num in animal_data['channel_data']:
                        target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in animal_data['preprocessed_data'].columns:
                            smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            animal_data['preprocessed_data'][smoothed_col] = savgol_filter(
                                animal_data['preprocessed_data'][target_col], window_size, poly_order)
            
            fig, ax = plt.subplots()
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_animals)))
            for idx, animal_data in enumerate(selected_animals):
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
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        try:
            model_type = self.baseline_model.get()
            
            for animal_data in selected_animals:
                if animal_data.get('preprocessed_data') is None:
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
            
            fig, ax = plt.subplots()
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_animals)))
            for idx, animal_data in enumerate(selected_animals):
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
            
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        try:
            for animal_data in selected_animals:
                if animal_data.get('preprocessed_data') is None:
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
                        fitted_ref_col = f"CH{channel_num}_fitted_ref"
                        animal_data['preprocessed_data'][fitted_ref_col] = predicted_signal
            
            fig, ax = plt.subplots()
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_animals)))
            for idx, animal_data in enumerate(selected_animals):
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
        except Exception as e:
            messagebox.showerror("Error", f"Motion correction failed: {str(e)}")
            self.set_status("Motion correction failed")
    
    def apply_preprocessing(self):
        try:
            selected_animals = self.get_selected_animals()
            if not selected_animals:
                return
            
            if self.apply_smooth.get():
                self.smooth_data()
            if self.apply_baseline.get():
                self.baseline_correction()
            if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                self.motion_correction()
            
            messagebox.showinfo("Success", f"Preprocessing applied to {len(selected_animals)} animal(s)")
            self.set_status(f"Preprocessing applied to {len(selected_animals)} animal(s)")
                
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.set_status("Preprocessing failed")

    def calculate_and_plot_dff(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        try:
            for animal_data in selected_animals:
                if animal_data.get('preprocessed_data') is None:
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
                        target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                        if not target_col or target_col not in animal_data['preprocessed_data'].columns:
                            continue
                        
                        if f"CH{channel_num}_{self.target_signal_var.get()}_smoothed" in animal_data['preprocessed_data'].columns:
                            raw_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                        else:
                            raw_col = target_col
                        
                        raw_target = animal_data['preprocessed_data'][raw_col]
                        median_full = np.median(raw_target)
                        
                        if ref_wavelength == "410" and self.apply_baseline.get():
                            if f"CH{channel_num}_motion_corrected" in animal_data['preprocessed_data'].columns:
                                motion_corrected = animal_data['preprocessed_data'][f"CH{channel_num}_motion_corrected"]
                                dff_data = motion_corrected / median_full
                            else:
                                messagebox.showerror("Error", "Please apply the motion correction")
                        elif ref_wavelength == "410" and not self.apply_baseline.get():
                            if f"CH{channel_num}_fitted_ref" in animal_data['preprocessed_data'].columns:
                                fitted_ref = animal_data['preprocessed_data'][f"CH{channel_num}_fitted_ref"]
                                dff_data = (raw_target - fitted_ref) / fitted_ref
                            else:
                                messagebox.showerror("Error", "Please apply the motion correction")
                        elif ref_wavelength == "baseline" and self.apply_baseline.get():
                            if f"CH{channel_num}_baseline_pred" in animal_data['preprocessed_data'].columns:
                                baseline_fitted = animal_data['preprocessed_data'][f"CH{channel_num}_baseline_pred"]
                                baseline_median = np.median(raw_target[baseline_mask])
                                dff_data = (raw_target - baseline_fitted) / baseline_median
                            else:
                                messagebox.showerror("Error", "Please check the baseline correction")
                        elif ref_wavelength == "baseline" and not self.apply_baseline.get():
                            baseline_median = np.median(raw_target[baseline_mask])
                            dff_data = (raw_target - baseline_median) / baseline_median
                        
                        dff_col = f"CH{channel_num}_dff"
                        animal_data['preprocessed_data'][dff_col] = dff_data
                        animal_data['dff_data'][channel_num] = dff_data
            
            self.set_status("ΔF/F calculated")
        except Exception as e:
            messagebox.showerror("Error", f"ΔF/F calculation failed: {str(e)}")
            self.set_status("ΔF/F calculation failed")
    
    def calculate_and_plot_zscore(self):
        try:
            selected_animals = self.get_selected_animals()
            if not selected_animals:
                return
            
            for animal_data in selected_animals:
                if animal_data.get('dff_data') is None or not animal_data.get('active_channels'):
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
            
            self.set_status("Z-score calculated")
        except Exception as e:
            messagebox.showerror("Error", f"Z-score calculation failed: {str(e)}")
            self.set_status("Z-score calculation failed")
    
    def get_available_event_types(self):
        """Get available event types from selected animals (excluding type 0)"""
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return []
        
        event_types = set()
        for animal_data in selected_animals:
            if 'event_data' in animal_data:
                types = animal_data['event_data']['Event Type'].unique()
                event_types.update([t for t in types if t != 0])
        
        return sorted(list(event_types))
    
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
            
            start_time = event.get('start_absolute', event['start_time'])
            end_time = event.get('end_absolute', event['end_time'])
            
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

    def compute_event_zscore_from_dff(self, dff_data, time_data, event_start, event_end, pre_window, post_window, post_flag):
        event_window_start = event_start - pre_window
        if post_flag is True:
            event_window_end = event_end + post_window
        else:
            event_window_end = event_start + post_window
        
        time_data = np.asarray(time_data, dtype=float)
        dff_data = np.asarray(dff_data, dtype=float)

        window_mask = (time_data >= event_window_start) & (time_data <= event_window_end)
        window_time = time_data[window_mask]
        window_dff = dff_data[window_mask]
        
        valid_mask = ~np.isnan(window_dff) & ~np.isnan(window_time)
        window_time = window_time[valid_mask]
        window_dff = window_dff[valid_mask]

        if len(window_dff) == 0:
            return np.array([]), np.array([]), None
        
        baseline_mask = (window_time >= event_window_start) & (window_time < event_start)
        baseline_dff = window_dff[baseline_mask]
        
        if len(baseline_dff) < 2:
            return window_time - event_start, np.full(len(window_dff), np.nan), None
        
        baseline_mean = np.mean(baseline_dff)
        baseline_std = np.std(baseline_dff)
        
        if baseline_std == 0:
            baseline_std = 1e-6
        
        zscores = (window_dff - baseline_mean) / baseline_std
        
        return window_time - event_start, zscores, (baseline_mean, baseline_std)

    def plot_event_related_activity(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        if not all('zscore_data' in a for a in selected_animals):
            self.calculate_and_plot_zscore()

        try:
            # Get spacing parameters
            spacing_dialog = tk.Toplevel(self.root)
            spacing_dialog.title("Plot Spacing Settings")
            spacing_dialog.geometry("300x150")
            spacing_dialog.transient(self.root)
            spacing_dialog.grab_set()
            
            main_frame = ttk.Frame(spacing_dialog, padding=10)
            main_frame.pack(fill="both", expand=True)
            
            use_spacing_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(main_frame, text="Use spacing between lines", 
                          variable=use_spacing_var).pack(pady=5)
            
            spacing_frame = ttk.Frame(main_frame)
            spacing_frame.pack(pady=5)
            ttk.Label(spacing_frame, text="Spacing multiplier:").grid(row=0, column=0, padx=5)
            spacing_entry = ttk.Entry(spacing_frame, width=10)
            spacing_entry.insert(0, "0.6")
            spacing_entry.grid(row=0, column=1, padx=5)
            
            confirmed = tk.BooleanVar(value=False)
            
            use_spacing = True
            spacing_multiplier = 0.6

            def on_confirm():
                nonlocal use_spacing, spacing_multiplier
                use_spacing = use_spacing_var.get()
                try:
                    spacing_multiplier = float(spacing_entry.get())
                    confirmed.set(True)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter a valid number for spacing multiplier")
                    return
                spacing_dialog.destroy()
            
            ttk.Button(main_frame, text="Confirm", command=on_confirm).pack(pady=10)
            
            spacing_dialog.wait_window()
            
            if not confirmed.get():
                return
            
            # Get available event types
            event_types = self.get_available_event_types()
            if not event_types:
                messagebox.showwarning("No Events", "No events found in selected animals")
                return
            
            # Select event type
            event_type_dialog = tk.Toplevel(self.root)
            event_type_dialog.title("Select Event Type")
            event_type_dialog.geometry("250x200")
            event_type_dialog.transient(self.root)
            event_type_dialog.grab_set()
            
            main_frame = ttk.Frame(event_type_dialog, padding=10)
            main_frame.pack(fill="both", expand=True)
            
            ttk.Label(main_frame, text="Select Event Type:").pack(pady=5)
            
            selected_type = tk.IntVar(value=event_types[0])
            for et in event_types:
                ttk.Radiobutton(main_frame, text=f"Type {et}", 
                              variable=selected_type, value=et).pack(anchor="w", padx=20)
            
            ttk.Button(main_frame, text="Confirm", 
                      command=event_type_dialog.destroy).pack(pady=10)
            
            event_type_dialog.wait_window()
            
            selected_type_val = selected_type.get()
            
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            
            groups = {}
            for animal_data in selected_animals:
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Event-Related Activity")
            plot_window.geometry("1000x800")
            
            tab_control = ttk.Notebook(plot_window)
            
            for group_name, animal_list in groups.items():
                group_events = []
                for animal_data in animal_list:
                    if 'event_data' in animal_data:
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type_val]
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

                        for channel_num in animal_data['active_channels']:
                            if channel_num in animal_data['dff_data']:
                                dff_data = animal_data['dff_data'][channel_num]
                        
                                event_time_rel, event_zscore, _ = self.compute_event_zscore_from_dff(
                                        dff_data, time_data, start_time, end_time, pre_window, post_window, post_flag=True
                                    )
                                
                                if len(event_time_rel) > 0:
                                    event_responses[orig_idx].append((event_time_rel, event_zscore))
                
                if use_spacing:
                    all_zscores = []
                    for event_idx in event_responses:
                        for times, zscores in event_responses[event_idx]:
                            all_zscores.extend(zscores)
                    data_range = max(all_zscores) - min(all_zscores) if all_zscores else 5
                    y_spacing = data_range * spacing_multiplier
                else:
                    y_spacing = 0
                
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
                        events1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type_val]
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
            self.set_status("Event-related activity plotted in new window")
        except Exception as e:
            messagebox.showerror("Error", f"Event analysis failed: {str(e)}")
            self.set_status("Event analysis failed")

    def plot_heatmap(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        if not all('dff_data' in a for a in selected_animals):
            self.calculate_and_plot_dff()
            
        try:
            # Get available event types
            event_types = self.get_available_event_types()
            if not event_types:
                messagebox.showwarning("No Events", "No events found in selected animals")
                return
            
            # Select event type
            event_type_dialog = tk.Toplevel(self.root)
            event_type_dialog.title("Select Event Type")
            event_type_dialog.geometry("250x200")
            event_type_dialog.transient(self.root)
            event_type_dialog.grab_set()
            
            main_frame = ttk.Frame(event_type_dialog, padding=10)
            main_frame.pack(fill="both", expand=True)
            
            ttk.Label(main_frame, text="Select Event Type:").pack(pady=5)
            
            selected_type = tk.IntVar(value=event_types[0])
            for et in event_types:
                ttk.Radiobutton(main_frame, text=f"Type {et}", 
                              variable=selected_type, value=et).pack(anchor="w", padx=20)
            
            ttk.Button(main_frame, text="Confirm", 
                      command=event_type_dialog.destroy).pack(pady=10)
            
            event_type_dialog.wait_window()
            
            selected_type_val = selected_type.get()
            
            pre_window = float(self.pre_event.get())
            post_window = float(self.post_event.get())
            
            groups = {}
            for animal_data in selected_animals:
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Activity Heatmap")
            plot_window.geometry("1000x800")
            
            tab_control = ttk.Notebook(plot_window)
            
            for group_name, animal_list in groups.items():
                group_event_list = []
                valid_animal_count = 0
                
                for animal_data in animal_list:
                    if 'event_data' not in animal_data or 'dff_data' not in animal_data:
                        continue
                    
                    events = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type_val]
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
                
                all_activities = []
                row_labels = []
                
                for animal_data, events in group_event_list:
                    filtered_events = events.iloc[selected_indices]
                    if filtered_events.empty:
                        continue
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['dff_data']:
                            dff_data = animal_data['dff_data'][channel_num]
                            
                            for event_idx, (_, event) in enumerate(filtered_events.iterrows()):
                                if animal_data.get('event_time_absolute', False):
                                    start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                    end_time = event['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    start_time = event['start_time']
                                    end_time = event['end_time']

                                event_end = end_time + post_window
                                
                                event_time_relative, event_zscore, _ = self.compute_event_zscore_from_dff(
                                        dff_data, time_data, start_time, end_time, pre_window, post_window, post_flag=True
                                    )
                                
                                interp_time = np.linspace(-pre_window, event_end - start_time, 200)
                                if len(event_time_relative) > 1:
                                    animal_id = animal_data['animal_id']
                                    interp_zscore = np.interp(interp_time, event_time_relative, event_zscore)
                                    all_activities.append(interp_zscore)
                                    row_labels.append(f"{animal_id} CH{channel_num} Event {event_idx+1}")
                
                if not all_activities:
                    continue
                    
                activity_matrix = np.array(all_activities)
                
                colors = ["blue", "white", "red"]
                cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors, N=256)
                
                center = np.median(activity_matrix)
                norm = TwoSlopeNorm(vmin=activity_matrix.min(), vcenter=center, vmax=activity_matrix.max())

                im = ax.imshow(activity_matrix, aspect='auto', cmap=cmap, norm=norm, 
                            extent=[-pre_window, event_end - start_time, 0, len(all_activities)])
                
                ax.axvline(0, color='k', linestyle='--', linewidth=1)

                if len(group_event_list) > 0:
                    first_event = group_event_list[0][1].iloc[selected_indices[0]]
                    first_animal_data = group_event_list[0][0]
                    if first_animal_data.get('event_time_absolute', False):
                        event_duration = first_event['end_absolute'] - first_event['start_absolute']
                    else:
                        event_duration = first_event['end_time'] - first_event['start_time']
                    ax.axvline(event_duration, color='k', linestyle='--', linewidth=1)

                    for animal_data in animal_list:
                        if 'event_data' in animal_data:
                            events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                            events1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type_val]
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
                ax.set_title(f"Group: {group_name} - Event-Related Activity Heatmap ({len(selected_indices)} events)")
                
                fig.colorbar(im, ax=ax, label="Z-Score")
                fig.tight_layout()
                canvas.draw()
            
            tab_control.pack(expand=1, fill="both")
            self.set_status("Heatmap plotted in new window")
        except Exception as e:
            messagebox.showerror("Error", f"Heatmap creation failed: {str(e)}")
            self.set_status("Heatmap failed")
    
    def plot_experiment_related_activity_and_export_statistic_results(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return

        try:
            # Get available event types
            event_types = self.get_available_event_types()
            if not event_types:
                messagebox.showwarning("No Events", "No events found in selected animals")
                return
            
            # Select event type
            event_type_dialog = tk.Toplevel(self.root)
            event_type_dialog.title("Select Event Type")
            event_type_dialog.geometry("250x200")
            event_type_dialog.transient(self.root)
            event_type_dialog.grab_set()
            
            main_frame = ttk.Frame(event_type_dialog, padding=10)
            main_frame.pack(fill="both", expand=True)
            
            ttk.Label(main_frame, text="Select Event Type:").pack(pady=5)
            
            selected_type = tk.IntVar(value=event_types[0])
            for et in event_types:
                ttk.Radiobutton(main_frame, text=f"Type {et}", 
                            variable=selected_type, value=et).pack(anchor="w", padx=20)
            
            selected_type_val = selected_type.get()

            ttk.Button(main_frame, text="Confirm", 
                    command=event_type_dialog.destroy).pack(pady=10)
            
            event_type_dialog.wait_window()
            
            # Check if any animal has event type 2
            has_type2 = False
            for animal_data in selected_animals:
                if 'event_data' in animal_data:
                    if 2 in animal_data['event_data']['Event Type'].values:
                        has_type2 = True
                        break
            
            # Create time window dialog
            window_dialog = tk.Toplevel(self.root)
            window_dialog.title("Time Window Settings")
            window_dialog.geometry("400x300")
            window_dialog.transient(self.root)
            window_dialog.grab_set()
            
            time_values = {
                'pre_start': tk.StringVar(value="2"),
                'post_start': tk.StringVar(value="2"),
                'post_end': tk.StringVar(value="2")
            }
            
            if has_type2:
                time_values['pre_type2'] = tk.StringVar(value="2")
            else:
                time_values['pre_end'] = tk.StringVar(value="2")
            
            main_frame = ttk.Frame(window_dialog, padding=10)
            main_frame.pack(fill="both", expand=True)
            
            grid_frame = ttk.Frame(main_frame)
            grid_frame.pack(pady=10)
            
            ttk.Label(grid_frame, text="Time Windows (seconds):").grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
            
            entries = {}
            
            ttk.Label(grid_frame, text="Pre Event Type 1 Start Time:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            entries['pre_start'] = ttk.Entry(grid_frame, width=10, textvariable=time_values['pre_start'])
            entries['pre_start'].grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(grid_frame, text="Post Event Type 1 Start Time:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            entries['post_start'] = ttk.Entry(grid_frame, width=10, textvariable=time_values['post_start'])
            entries['post_start'].grid(row=2, column=1, padx=5, pady=5)
            
            if has_type2:
                ttk.Label(grid_frame, text="Pre Event Type 2 Start Time:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
                entries['pre_type2'] = ttk.Entry(grid_frame, width=10, textvariable=time_values['pre_type2'])
                entries['pre_type2'].grid(row=3, column=1, padx=5, pady=5)
            else:
                ttk.Label(grid_frame, text="Pre Event Type 1 End Time:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
                entries['pre_end'] = ttk.Entry(grid_frame, width=10, textvariable=time_values['pre_end'])
                entries['pre_end'].grid(row=3, column=1, padx=5, pady=5)
            
            ttk.Label(grid_frame, text="Post Event Type 1 End Time:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
            entries['post_end'] = ttk.Entry(grid_frame, width=10, textvariable=time_values['post_end'])
            entries['post_end'].grid(row=4, column=1, padx=5, pady=5)
            
            export_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(grid_frame, text="Export curve statistics", 
                        variable=export_var).grid(row=5, column=0, columnspan=2, pady=10, sticky="w")
            
            confirmed = tk.BooleanVar(value=False)
            
            def on_confirm():
                global time_window_values, export_stats_value
                time_window_values = {}
                for key, var in time_values.items():
                    time_window_values[key] = var.get()
                export_stats_value = export_var.get()
                confirmed.set(True)
                window_dialog.destroy()
            
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="Confirm", command=on_confirm).pack()
            
            def on_closing():
                confirmed.set(False)
                window_dialog.destroy()
            
            window_dialog.protocol("WM_DELETE_WINDOW", on_closing)
            
            window_dialog.wait_window()
            
            if not confirmed.get():
                return
            
            time_windows = {
                'pre_start': float(time_window_values['pre_start']),
                'post_start': float(time_window_values['post_start']),
                'post_end': float(time_window_values['post_end'])
            }
            
            if has_type2:
                time_windows['pre_type2'] = float(time_window_values['pre_type2'])
            else:
                time_windows['pre_end'] = float(time_window_values['pre_end'])
            
            export_stats = export_stats_value
            
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Experiment-Related Activity")
            plot_window.geometry("1000x800")
            
            tab_control = ttk.Notebook(plot_window)
            groups = {}
            
            for animal_data in selected_animals:
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            all_statistics = []
            
            for group_name, animal_list in groups.items():
                group_event_list = []
                valid_animal_count = 0
                
                for animal_data in animal_list:
                    if 'event_data' not in animal_data or 'dff_data' not in animal_data:
                        continue
                    
                    events = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type_val]
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
                        if channel_num not in animal_data['dff_data']:
                            continue
                        
                        dff_data = animal_data['dff_data'][channel_num]
                        
                        for event_idx, (_, event) in enumerate(filtered_events.iterrows()):
                            if animal_data.get('event_time_absolute', False):
                                start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                end_time = event['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start_time = event['start_time']
                                end_time = event['end_time']
                            
                            duration = end_time - start_time
                            
                            event_time_relative, event_dff, _ = self.compute_event_zscore_from_dff(
                                    dff_data, time_data, start_time, end_time, 
                                    max(time_windows['pre_start'], time_windows.get('pre_end', 0), 
                                        time_windows.get('pre_type2', 0)), 
                                    time_windows['post_end'], post_flag=True
                                )
                            
                            if len(event_time_relative) > 0:
                                all_responses.append((event_time_relative, event_dff))
                                
                                # Calculate statistics if requested
                                if export_stats:
                                    # Pre start window
                                    mask_pre_start = (event_time_relative >= -time_windows['pre_start']) & (event_time_relative < 0)
                                    pre_start_data = event_dff[mask_pre_start]
                                    
                                    # Post start window
                                    mask_post_start = (event_time_relative >= 0) & (event_time_relative <= time_windows['post_start'])
                                    post_start_data = event_dff[mask_post_start]
                                    
                                    # Post end window
                                    mask_post_end = (event_time_relative >= duration) & (event_time_relative <= duration + time_windows['post_end'])
                                    post_end_data = event_dff[mask_post_end]
                                    
                                    stats_entry = {
                                        'Group': group_name,
                                        'Animal ID': animal_data['animal_id'],
                                        'Channel': channel_num,
                                        'Event Index': event_idx + 1,
                                        'Pre Start Mean': np.mean(pre_start_data) if len(pre_start_data) > 0 else np.nan,
                                        'Pre Start Max': np.max(pre_start_data) if len(pre_start_data) > 0 else np.nan,
                                        'Pre Start Min': np.min(pre_start_data) if len(pre_start_data) > 0 else np.nan,
                                        'Pre Start AUC': np.trapz(pre_start_data, event_time_relative[mask_pre_start]) if len(pre_start_data) > 0 else np.nan,
                                        'Post Start Mean': np.mean(post_start_data) if len(post_start_data) > 0 else np.nan,
                                        'Post Start Max': np.max(post_start_data) if len(post_start_data) > 0 else np.nan,
                                        'Post Start Min': np.min(post_start_data) if len(post_start_data) > 0 else np.nan,
                                        'Post Start AUC': np.trapz(post_start_data, event_time_relative[mask_post_start]) if len(post_start_data) > 0 else np.nan,
                                        'Post End Mean': np.mean(post_end_data) if len(post_end_data) > 0 else np.nan,
                                        'Post End Max': np.max(post_end_data) if len(post_end_data) > 0 else np.nan,
                                        'Post End Min': np.min(post_end_data) if len(post_end_data) > 0 else np.nan,
                                        'Post End AUC': np.trapz(post_end_data, event_time_relative[mask_post_end]) if len(post_end_data) > 0 else np.nan,
                                    }
                                    
                                    if has_type2:
                                        # Find type 2 events within this type 1 event
                                        events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                                        for _, event2 in events2.iterrows():
                                            if animal_data.get('event_time_absolute', False):
                                                start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                                end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                                            else:
                                                start2 = event2['start_time']
                                                end2 = event2['end_time']
                                            
                                            if start2 >= start_time and end2 <= end_time:
                                                # Pre type 2 window
                                                rel_start2 = start2 - start_time
                                                mask_pre_type2 = (event_time_relative >= rel_start2 - time_windows['pre_type2']) & (event_time_relative < rel_start2)
                                                pre_type2_data = event_dff[mask_pre_type2]
                                                
                                                # During type 2 window
                                                rel_end2 = end2 - start_time
                                                mask_type2 = (event_time_relative >= rel_start2) & (event_time_relative <= rel_end2)
                                                type2_data = event_dff[mask_type2]
                                                
                                                stats_entry['Pre Type2 Mean'] = np.mean(pre_type2_data) if len(pre_type2_data) > 0 else np.nan
                                                stats_entry['Pre Type2 Max'] = np.max(pre_type2_data) if len(pre_type2_data) > 0 else np.nan
                                                stats_entry['Pre Type2 Min'] = np.min(pre_type2_data) if len(pre_type2_data) > 0 else np.nan
                                                stats_entry['Pre Type2 AUC'] = np.trapz(pre_type2_data, event_time_relative[mask_pre_type2]) if len(pre_type2_data) > 0 else np.nan
                                                stats_entry['Type2 Mean'] = np.mean(type2_data) if len(type2_data) > 0 else np.nan
                                                stats_entry['Type2 Max'] = np.max(type2_data) if len(type2_data) > 0 else np.nan
                                                stats_entry['Type2 Min'] = np.min(type2_data) if len(type2_data) > 0 else np.nan
                                                stats_entry['Type2 AUC'] = np.trapz(type2_data, event_time_relative[mask_type2]) if len(type2_data) > 0 else np.nan
                                                break
                                    else:
                                        # Pre end window
                                        mask_pre_end = (event_time_relative >= duration - time_windows['pre_end']) & (event_time_relative < duration)
                                        pre_end_data = event_dff[mask_pre_end]
                                        
                                        stats_entry['Pre End Mean'] = np.mean(pre_end_data) if len(pre_end_data) > 0 else np.nan
                                        stats_entry['Pre End Max'] = np.max(pre_end_data) if len(pre_end_data) > 0 else np.nan
                                        stats_entry['Pre End Min'] = np.min(pre_end_data) if len(pre_end_data) > 0 else np.nan
                                        stats_entry['Pre End AUC'] = np.trapz(pre_end_data, event_time_relative[mask_pre_end]) if len(pre_end_data) > 0 else np.nan
                                    
                                    all_statistics.append(stats_entry)
                
                if not all_responses:
                    continue
                
                fps = fps_fiber if 'fps_fiber' in globals() else 10
                max_time = max(max(t) for t, _ in all_responses)
                common_time = np.linspace(min(t[0] for t, _ in all_responses), max_time,
                                        int((max_time - min(t[0] for t, _ in all_responses)) * fps))
                all_interp = []
                
                for time_rel, dff in all_responses:
                    if len(time_rel) > 1:
                        interp_dff = np.interp(common_time, time_rel, dff)
                        all_interp.append(interp_dff)
                
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
                if len(group_event_list) > 0:
                    first_animal_data = group_event_list[0][0]
                    first_event = group_event_list[0][1].iloc[selected_indices[0]]
                    if first_animal_data.get('event_time_absolute', False):
                        duration = first_event['end_absolute'] - first_event['start_absolute']
                    else:
                        duration = first_event['end_time'] - first_event['start_time']
                    ax.axvline(duration, color='k', linestyle='--', linewidth=1, label='Event End')
                    ax.axvspan(0, duration, color='yellow', alpha=0.2, label='Event Duration')
                
                for animal_data in animal_list:
                    if 'event_data' not in animal_data:
                        continue
                    
                    events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                    events1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == selected_type_val]
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
                ax.set_ylabel("ΔF/F")
                ax.set_title(f"Group: {group_name}\nExperiment-Related Activity "
                            f"({len(all_interp)} events from {valid_animal_count} animals)")
                ax.grid(False)
                ax.legend()
                
                canvas.draw()
            
            tab_control.pack(expand=1, fill="both")
            
            # Export statistics if requested
            if export_stats and all_statistics:
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    title="Save Curve Statistics"
                )
                
                if save_path:
                    df = pd.DataFrame(all_statistics)
                    df.to_csv(save_path, index=False, float_format='%.10f')
                    messagebox.showinfo("Success", f"Statistics exported to:\n{save_path}")
            
            self.set_status("Experiment-related activity plotted in new window")
        except Exception as e:
            messagebox.showerror("Error", f"Experiment-related analysis failed: {str(e)}")
            self.set_status("Experiment analysis failed")

class FreezingAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Freezing Analyzer")
        self.dlc_results_path = ""
        self.events_path = ""
        self.freezing_data = None
        self.window_freeze = None
        self.fps = 30
        self.raw_speed = None
        self.interp_speed = None
        self.smooth_speed = None
        self.freeze_threshold = None
        self.freeze_idx = None
        self.dlc_freeze_status = None
        self.multi_animal_results = {}
        self.cam_id = None
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)

        animal_frame = ttk.LabelFrame(control_panel, text="Animal Selection")
        animal_frame.pack(fill="x", padx=5, pady=5)
        
        list_frame = ttk.Frame(animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Single Import", command=self.add_single_file).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Multi Import", command=self.import_multi_animal_data).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        analysis_frame = ttk.LabelFrame(control_panel, text="Freezing Analysis")
        analysis_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(analysis_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Plot Raster", command=self.plot_multi_animal_raster).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Plot Freezing Timeline", command=self.plot_freezing_windows).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Export Freezing", command=self.export_freezing_csv).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Compare Raw vs Cleaned", command=self.plot_compare_raw_cleaning).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            fiber_frame.columnconfigure(0, weight=1)
            fiber_frame.columnconfigure(1, weight=1)
            
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            ttk.Button(fiber_frame, text="Plot Freezing w/Fiber", command=self.plot_freezing_with_fiber).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
    
    def show_step1(self):
        self.set_status("Step 1: Data Loading")
    
    def show_step2(self):
        self.set_status("Step 2: Preprocessing")

        if not hasattr(self, 'step2_frame') or not self.step2_frame:
            self.step2_frame = ttk.LabelFrame(self.control_frame, text="Preprocessing")
            self.step2_frame.pack(fill="x", padx=5, pady=5)
        
        self.step2_frame.pack(fill="x", padx=5, pady=5)
        self.step2_frame.update_idletasks()
        
        for widget in self.step2_frame.winfo_children():
            widget.destroy()
        
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
    
    def show_step3(self):
        self.set_status("Step 3: ΔF/F & Z-score")
        
        if not hasattr(self, 'step3_frame') or not self.step3_frame:
            self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
            self.step3_frame.pack(fill="x", padx=5, pady=5)
        
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        self.step3_frame.update_idletasks()
        
        for widget in self.step3_frame.winfo_children():
            widget.destroy()

        self.step3_frame.columnconfigure(0, weight=1)
        self.step3_frame.columnconfigure(1, weight=1)
        
        ttk.Button(self.step3_frame, text="Calculate & plot ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate & plot Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")

        if not hasattr(self, 'step4_frame') or not self.step4_frame:
            self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
            self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        self.step4_frame.update_idletasks()
        
        for widget in self.step4_frame.winfo_children():
            widget.destroy()
        
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
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text="Plot Activity1", 
                  command=self.plot_event_related_activity).grid(row=0, column=0, sticky="ew")

        ttk.Button(btn_frame, text="Plot Heatmap", 
                  command=self.plot_heatmap).grid(row=0, column=1, sticky="ew")
        
        ttk.Button(btn_frame, text="Plot Activity2 & Statistic", 
                  command=self.plot_experiment_related_activity_and_export_statistic_results).grid(row=1, column=0, sticky="ew")
    
    def analyze_multi_animal(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        self.set_status("Analyzing selected animals...")
        self.multi_animal_results = {}
        
        for animal_data in selected_animals:
            animal_id = animal_data['animal_id']
            group = animal_data['group']
            
            if group not in self.multi_animal_results:
                self.multi_animal_results[group] = []
            
            try:
                if 'dlc_data' not in animal_data or 'event_data' not in animal_data:
                    self.set_status(f"Skipping {animal_id}: missing DLC or event data")
                    continue

                freezing_result = self.compute_freezing_for_animal(animal_data)
                
                if freezing_result:
                    animal_data.update(freezing_result)
                    self.multi_animal_results[group].append(animal_data)
                    self.set_status(f"Analyzed {animal_id}")
                else:
                    self.set_status(f"Failed to analyze {animal_id}")
                    
            except Exception as e:
                self.set_status(f"Error analyzing {animal_id}: {str(e)}")
                print(f"Error analyzing {animal_id}: {traceback.format_exc()}")
        
        if any(self.multi_animal_results.values()):
            self.set_status("Analysis completed")
            messagebox.showinfo("Success", f"Analyzed {sum(len(v) for v in self.multi_animal_results.values())} animals")
        else:
            self.set_status("Analysis failed - no valid results")
            messagebox.showwarning("Warning", "No valid animals were analyzed")
    
    def compute_freezing_for_animal(self, animal_data):
        try:
            dlc_result = animal_data['dlc_data']
            event_data = animal_data['event_data']
            
            scorer = dlc_result.columns.levels[0][0]
            left_ear = 'left_ear'
            right_ear = 'right_ear'
            tail_base = 'tail_base'
            mid_back = 'mid_back'

            le_x = dlc_result.loc[:, (scorer, left_ear, 'x')].values.astype(float)
            le_y = dlc_result.loc[:, (scorer, left_ear, 'y')].values.astype(float)
            le_r = dlc_result.loc[:, (scorer, left_ear, 'likelihood')].values.astype(float)
            
            re_x = dlc_result.loc[:, (scorer, right_ear, 'x')].values.astype(float)
            re_y = dlc_result.loc[:, (scorer, right_ear, 'y')].values.astype(float)
            re_r = dlc_result.loc[:, (scorer, right_ear, 'likelihood')].values.astype(float)
            
            tb_x = dlc_result.loc[:, (scorer, tail_base, 'x')].values.astype(float)
            tb_y = dlc_result.loc[:, (scorer, tail_base, 'y')].values.astype(float)
            tb_r = dlc_result.loc[:, (scorer, tail_base, 'likelihood')].values.astype(float)

            mb_x = dlc_result.loc[:, (scorer, mid_back, 'x')].values.astype(float)
            mb_y = dlc_result.loc[:, (scorer, mid_back, 'y')].values.astype(float)
            mb_r = dlc_result.loc[:, (scorer, mid_back, 'likelihood')].values.astype(float)

            def interpolate_low_reliability(values, reliability, threshold=0.99):
                valid_mask = reliability >= threshold
                indices = np.arange(len(values))
                
                if np.sum(valid_mask) > 1:
                    interp_func = interp1d(
                        indices[valid_mask], 
                        values[valid_mask], 
                        kind='linear', 
                        fill_value="extrapolate"
                    )
                    interpolated = interp_func(indices)
                    return interpolated
                else:
                    return values

            le_x = interpolate_low_reliability(le_x, le_r, 0.99)
            le_y = interpolate_low_reliability(le_y, le_r, 0.99)
            re_x = interpolate_low_reliability(re_x, re_r, 0.99)
            re_y = interpolate_low_reliability(re_y, re_r, 0.99)
            tb_x = interpolate_low_reliability(tb_x, tb_r, 0.99)
            tb_y = interpolate_low_reliability(tb_y, tb_r, 0.99)

            centroid_x = (le_x + re_x + tb_x) / 3
            centroid_y = (le_y + re_y + tb_y) / 3

            dx = np.diff(centroid_x)
            dy = np.diff(centroid_y)
            dist = np.sqrt(dx**2 + dy**2)
            dist = np.insert(dist, 0, 0)
            raw_dist = dist

            dist[dist > 200] = np.nan
            valid = ~np.isnan(dist)
            dist_interp = np.interp(np.arange(len(dist)), np.flatnonzero(valid), dist[valid])

            kernel = np.ones(15) / 15
            speed_smooth = np.convolve(dist_interp, kernel, mode='same')

            threshold = 1.20
            freeze_flag = speed_smooth < threshold

            groups = groupby(enumerate(freeze_flag), key=lambda x: x[1])
            freeze_idx = []
            for k, g in groups:
                g = list(g)
                if k and len(g) >= 15:
                    idxs = [i for i, _ in g]
                    freeze_idx.extend(idxs)

            window_sec = 30
            window_size = int(self.fps * window_sec)

            start_time = event_data['start_time'].iloc[0]
            end_time = event_data['end_time'].iloc[-1]
            start_frame = int(start_time * self.fps)
            end_frame = int(end_time * self.fps)

            segment = speed_smooth[start_frame:end_frame]

            result_vector = []
            time_vector = []

            for i in range(0, len(segment), window_size):
                window_idx = np.arange(start_frame + i, min(start_frame + i + window_size, end_frame))
                count = np.sum(np.isin(window_idx, freeze_idx))
                result_vector.append(count)
                time_vector.append((start_frame + i + min(len(window_idx) // 2, window_size // 2)) / self.fps)

            df = pd.DataFrame({
                'time': time_vector,
                'freeze_sec': np.array(result_vector) / self.fps
            })

            dlc_freeze_status = np.zeros(len(speed_smooth), dtype=int)
            dlc_freeze_status[freeze_idx] = 1

            return {
                'window_freeze': df,
                'raw_speed': raw_dist,
                'interp_speed': dist_interp,
                'smooth_speed': speed_smooth,
                'freeze_threshold': threshold,
                'freeze_idx': freeze_idx,
                'dlc_freeze_status': dlc_freeze_status
            }
            
        except Exception as e:
            self.set_status(f"Error computing freezing: {str(e)}")
            print(f"Error computing freezing: {traceback.format_exc()}")
            return None
    
    def plot_multi_animal_raster(self):
        self.analyze_multi_animal()
        if not self.multi_animal_results:
            messagebox.showwarning("No Data", "Please select animals first")
            return
        
        try:
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Freezing Raster")
            plot_window.geometry("1000x800")
            
            tab_control = ttk.Notebook(plot_window)
            
            for group_name, animal_list in self.multi_animal_results.items():
                if not animal_list:
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
                
                y_ticks = []
                y_labels = []
                y_pos = 0
                
                for animal_data in animal_list:
                    animal_id = animal_data['animal_id']
                    
                    if 'dlc_freeze_status' not in animal_data:
                        continue
                    
                    dlc_freeze_status = animal_data['dlc_freeze_status']
                    event_data = animal_data['event_data']
                    
                    y_ticks.append(y_pos)
                    y_labels.append(animal_id)
                    
                    time_values = np.arange(len(dlc_freeze_status)) / self.fps
                        
                    for _, event in event_data.iterrows():
                        event_type = event['Event Type']
                        color = "#ffffff" if event_type == 0 else "#0400fc"

                        ax.add_patch(plt.Rectangle(
                            (event['start_time'], y_pos - 0.4),
                            event['end_time'] - event['start_time'],
                            0.8, color=color, alpha=0.5
                        ))
                    
                    freezing_times = time_values[dlc_freeze_status == 1]
                    for t in freezing_times:
                        ax.vlines(t, y_pos - 0.4, y_pos + 0.4, 
                                 color='black', alpha=0.5, linewidth=0.5)
                    
                    y_pos += 1
                
                if y_ticks:
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Animals")
                    ax.set_title(f"Freezing Raster Plot - {group_name}")
                    ax.grid(False)
                
                canvas.draw()
            
            if tab_control.tabs():
                tab_control.pack(expand=1, fill="both")
                self.set_status("Raster plotted")
            else:
                plot_window.destroy()
                messagebox.showwarning("No Data", "No valid data to plot")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raster: {str(e)}")
            self.set_status("Raster plot failed")
            print(f"Error plotting raster: {traceback.format_exc()}")
    
    def plot_freezing_windows(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        self.analyze_multi_animal()
        
        try:
            self.set_status("Plotting freezing timeline...")
            
            groups = {}
            for group_name, animal_list in self.multi_animal_results.items():
                groups[group_name] = animal_list
            
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Freezing Timeline")
            plot_window.geometry("1000x800")
            
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
                
                for idx, animal_data in enumerate(animal_list):
                    if 'window_freeze' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    window_freeze = animal_data['window_freeze']
                    event_data = animal_data['event_data']
                    
                    x = window_freeze['time'].values
                    y = window_freeze['freeze_sec'].values
                    
                    ax.plot(x, y, marker='o', color=colors[idx], label=animal_id)
                
                if len(animal_list) > 0 and 'event_data' in animal_list[0]:
                    event_data = animal_list[0]['event_data']
                    for _, row in event_data.iterrows():
                        color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                        ax.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Freezing Duration (s) per 30s window")
                ax.set_title(f"Freezing Timeline with Events - {group_name}")
                ax.grid(False)
                ax.legend()
                
                canvas.draw()
            
            tab_control.pack(expand=1, fill="both")
            self.set_status("Freezing timeline plotted.")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot freezing timeline: {str(e)}")
            self.set_status("Plot failed.")

    def plot_compare_raw_cleaning(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        self.analyze_multi_animal()
        
        for animal_data in selected_animals:
            if 'raw_speed' not in animal_data:
                continue
            
            animal_id = animal_data['animal_id']
            raw_speed = animal_data['raw_speed']
            interp_speed = animal_data['interp_speed']
            smooth_speed = animal_data['smooth_speed']
            freeze_threshold = animal_data['freeze_threshold']
            freeze_idx = animal_data['freeze_idx']
            event_data = animal_data['event_data']
            
            x_time = np.arange(len(raw_speed)) / 30.0

            raw_speed_plot = raw_speed.copy()
            valid = ~np.isnan(raw_speed_plot)
            if valid.sum() >= 2:
                raw_speed_plot = np.interp(np.arange(len(raw_speed_plot)), np.flatnonzero(valid), raw_speed_plot[valid])
            else:
                raw_speed_plot[:] = 0

            plt.figure(figsize=(12, 4))

            for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
                state_rows = event_data[event_data['Event Type'] == state_code]
                for _, row in state_rows.iterrows():
                    plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)

            plt.plot(x_time, raw_speed, alpha=0.6, label='Raw Speed')
            plt.plot(x_time, interp_speed, alpha=0.6, label='Interpolated')
            plt.plot(x_time, smooth_speed, linewidth=1.5, label='Smoothed')
            plt.axhline(freeze_threshold, linestyle='--', color="#ff0000ff", label='Threshold')

            in_freeze = np.zeros_like(smooth_speed)
            in_freeze[freeze_idx] = 1
            plt.fill_between(x_time, 0, max(smooth_speed), where=in_freeze > 0, color="#ff0000ff", alpha=0.3, label='Freezing')
            plt.xlabel("Time (s)")
            plt.ylabel("Speed (px/frame)")
            plt.title(f"Speed Trace - {animal_id}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def export_freezing_csv(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        self.analyze_multi_animal()
        
        all_freezing_data = []
        
        for group_name, animal_list in self.multi_animal_results.items():
            for animal_data in animal_list:
                if 'window_freeze' not in animal_data:
                    continue
                
                animal_id = animal_data['animal_id']
                window_freeze = animal_data['window_freeze'].copy()
                window_freeze['Animal ID'] = animal_id
                window_freeze['Group'] = group_name
                all_freezing_data.append(window_freeze)
        
        if not all_freezing_data:
            messagebox.showwarning("No Data", "No freezing data to export")
            return
        
        combined_df = pd.concat(all_freezing_data, ignore_index=True)
        combined_df = combined_df[['Animal ID', 'Group', 'time', 'freeze_sec']]
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            title="Export Freezing Results",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not save_path:
            return
        
        combined_df.to_csv(save_path, index=False)
        messagebox.showinfo("Exported", f"Freezing results saved to:\n{save_path}")
        self.set_status("Freezing results exported.")
    
    def plot_freezing_with_fiber(self):
        selected_animals = self.get_selected_animals()
        if not selected_animals:
            return
        
        self.analyze_multi_animal()
        
        try:
            self.fig.clear()
            
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = self.fig.add_subplot(gs[0])
            ax2 = self.fig.add_subplot(gs[1])
            
            all_signals = []
            common_time = None
            
            for animal_data in selected_animals:
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
            
            ax1.set_title('Freezing Analysis with Fiber Photometry')
            ax1.set_ylabel("Fiber Signal")
            ax1.grid(False)
            ax1.legend()
            
            y_ticks = []
            y_labels = []
            
            for idx, animal_data in enumerate(selected_animals):
                if 'dlc_freeze_status' not in animal_data:
                    continue
                
                dlc_freeze_status = animal_data['dlc_freeze_status']
                time_values = np.arange(len(dlc_freeze_status)) / self.fps
                freezing_times = time_values[dlc_freeze_status == 1]
                
                y_pos = idx
                y_ticks.append(y_pos)
                y_labels.append(animal_data['animal_id'])
                
                for t in freezing_times:
                    ax2.vlines(t, y_pos - 0.4, y_pos + 0.4, color='black', 
                              linewidth=0.5, alpha=0.7)
            
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels(y_labels)
            
            if len(selected_animals) > 0 and 'event_data' in selected_animals[0]:
                event_data = selected_animals[0]['event_data']
                event_time_absolute = selected_animals[0].get('event_time_absolute', False)
                
                if event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                    time_offset = selected_animals[0]['video_start_fiber']
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                    time_offset = 0
                
                for _, row in event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - time_offset
                    end_time = row[end_col] - time_offset
                    
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
            print(f"Error plotting freezing with fiber: {traceback.format_exc()}")

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
        
        animal_frame = ttk.LabelFrame(control_panel, text="Animal Selection")
        animal_frame.pack(fill="x", padx=5, pady=5)
        
        list_frame = ttk.Frame(animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Single Import", command=self.add_single_file).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Multi Import", command=self.import_multi_animal_data).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        analysis_frame = ttk.LabelFrame(control_panel, text="Pupil Analysis")
        analysis_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(analysis_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="Plot Pupil Distance", command=self.plot_pupil_distance).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Plot Combined Data", command=self.plot_combined_data).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            fiber_frame.columnconfigure(0, weight=1)
            fiber_frame.columnconfigure(1, weight=1)
            
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            ttk.Button(fiber_frame, text="Plot Pupil w/Fiber", command=self.plot_pupil_with_fiber).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

    def show_step1(self):
        self.set_status("Step 1: Data Loading")
    
    def show_step2(self):
        self.set_status("Step 2: Preprocessing")

        if not hasattr(self, 'step2_frame') or not self.step2_frame:
            self.step2_frame = ttk.LabelFrame(self.control_frame, text="Preprocessing")
            self.step2_frame.pack(fill="x", padx=5, pady=5)
        
        self.step2_frame.pack(fill="x", padx=5, pady=5)
        self.step2_frame.update_idletasks()
        
        for widget in self.step2_frame.winfo_children():
            widget.destroy()
        
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
    
    def show_step3(self):
        self.set_status("Step 3: ΔF/F & Z-score")
        
        if not hasattr(self, 'step3_frame') or not self.step3_frame:
            self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
            self.step3_frame.pack(fill="x", padx=5, pady=5)
        
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        self.step3_frame.update_idletasks()
        
        for widget in self.step3_frame.winfo_children():
            widget.destroy()
            
        self.step3_frame.columnconfigure(0, weight=1)
        self.step3_frame.columnconfigure(1, weight=1)
        
        ttk.Button(self.step3_frame, text="Calculate & plot ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate & plot Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")

        if not hasattr(self, 'step4_frame') or not self.step4_frame:
            self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
            self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        self.step4_frame.update_idletasks()
        
        for widget in self.step4_frame.winfo_children():
            widget.destroy()
        
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

    def h_AST2_readData(self, filename):
        header = {}
        data = None
        
        with open(filename, 'rb') as fid:
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
            
            fid.seek(0, 2)
            file_size = fid.tell()
            header_size = 4 + 2 + 2 + 4 + 4 + 20 + 40
            data_size = file_size - header_size
            
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

    def plot_pupil_distance(self):
        messagebox.showinfo("Info", "Pupil distance plotting for multi-animal mode is not yet implemented")
        self.set_status("Feature not available")
    
    def plot_combined_data(self):
        messagebox.showinfo("Info", "Combined data plotting for multi-animal mode is not yet implemented")
        self.set_status("Feature not available")
    
    def plot_pupil_with_fiber(self):
        messagebox.showinfo("Info", "Pupil with fiber plotting for multi-animal mode is not yet implemented")
        self.set_status("Feature not available")

def on_closing():
    root.quit()
    root.destroy()
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = ModeSelectionApp(root)
    root.mainloop()