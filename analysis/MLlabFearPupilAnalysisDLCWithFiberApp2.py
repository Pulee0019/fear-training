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
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
import fnmatch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import json
from scipy.interpolate import interp1d
from itertools import groupby
import traceback
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread")

invert = True

# Global channel memory
CHANNEL_MEMORY_FILE = "channel_memory.json"
channel_memory = {}

def load_channel_memory():
    """Load channel memory from file"""
    global channel_memory
    if os.path.exists(CHANNEL_MEMORY_FILE):
        try:
            with open(CHANNEL_MEMORY_FILE, 'r') as f:
                channel_memory = json.load(f)
        except:
            channel_memory = {}

def save_channel_memory():
    """Save channel memory to file"""
    try:
        with open(CHANNEL_MEMORY_FILE, 'w') as f:
            json.dump(channel_memory, f)
    except:
        pass

load_channel_memory()

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
        
        root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(root))
        root.mainloop()
    
    def on_closing(self, root):
        root.quit()
        root.destroy()

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
        self.selected_files = []
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)
        
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10, ipadx=5, ipady=5)
        
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
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
    
    def single_import(self):
        """Single animal import - select one animal folder"""
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
                'event_time_absolute': False,
                'active_channels': []
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
            self.selected_files.append(animal_data)
            self.multi_animal_data.append(animal_data)
            self.file_listbox.insert(tk.END, f"{animal_id} ({group})")
            
            if self.include_fiber and 'fiber_data' in animal_data:
                self.show_channel_selection_dialog()
            
            self.set_status(f"Added animal: {animal_id} ({group})")
            messagebox.showinfo("Success", f"Successfully added animal:\n{animal_id} ({group})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add single animal: {str(e)}")
            self.set_status("Add animal failed")

    def multi_import(self):
        """Multi-animal import - scan entire directory structure"""
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
                            self.selected_files.append(animal_data)
                            self.file_listbox.insert(tk.END, f"{animal_id} ({group})")

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

            if not self.selected_files:
                messagebox.showwarning("No Data", "No valid animal data found in the selected directory")
                self.set_status("No multi-animal data found")
            else:
                self.set_status(f"Found {len(self.selected_files)} animals")
                if self.include_fiber:
                    self.show_channel_selection_dialog()
                else:
                    messagebox.showinfo("Success", f"Found and processed {len(self.selected_files)} animals")
                    self.set_status(f"Imported {len(self.selected_files)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            self.set_status("Import failed")

    def clear_selected(self):
        """Clear selected animals from the list"""
        selected_indices = self.file_listbox.curselection()
        for index in sorted(selected_indices, reverse=True):
            animal_id = self.file_listbox.get(index)
            self.file_listbox.delete(index)
            
            if index < len(self.selected_files):
                animal_data = self.selected_files.pop(index)
                for i, data in enumerate(self.multi_animal_data):
                    if data['animal_id'] == animal_data['animal_id']:
                        self.multi_animal_data.pop(i)
                        break
        self.set_status(f"Cleared {len(selected_indices)} animals")
    
    def select_all(self):
        """Select all animals in the list"""
        self.file_listbox.selection_set(0, tk.END)
        self.set_status(f"Selected all {self.file_listbox.size()} animals")

    def load_fiber_data(self, file_path):
        """Load fiber photometry data"""
        try:
            fiber_data = pd.read_csv(file_path, skiprows=1, delimiter=',')
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
        """Show dialog for channel selection with memory"""
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
            
            # Load channel memory for this animal
            remembered_channels = channel_memory.get(animal_id, [1])
            
            for channel_num in sorted(animal_data['channel_data'].keys()):
                var = tk.BooleanVar(value=(channel_num in remembered_channels))
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
        """Toggle all channel selections"""
        for animal_vars in self.channel_vars.values():
            for var in animal_vars.values():
                var.set(select)

    def finalize_channel_selection(self, dialog):
        """Finalize channel selection and save to memory"""
        global channel_memory
        
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
                # Save to memory
                channel_memory[animal_id] = selected_channels
                self.set_video_markers(animal_data)

        dialog.destroy()
        save_channel_memory()
        messagebox.showinfo("Success", f"Processed {len(self.multi_animal_data)} animals")
        self.set_status(f"Imported {len(self.multi_animal_data)} animals")
    
    def set_video_markers(self, animal_data):
        """Set video markers for specific animal"""
        try:
            fiber_data = animal_data.get('fiber_data')
            event_data = animal_data.get('event_data')
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            
            if fiber_data is None or not active_channels:
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
            
            self.set_status(f"Video markers set for {animal_data['animal_id']}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set video markers: {str(e)}")
            self.set_status("Video markers failed")

    def plot_raw_data(self):
        """Plot raw fiber data for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        selected_animals = [self.multi_animal_data[i] for i in selected_indices]
        
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_animals)))
            for animal_idx, animal_data in enumerate(selected_animals):
                if 'fiber_cropped' not in animal_data:
                    continue
                    
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
            
            ax.set_title("Raw Fiber Photometry Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensity")
            ax.grid(False)
            
            if selected_animals and 'event_data' in selected_animals[0]:
                event_data = selected_animals[0]['event_data']
                event_time_absolute = selected_animals[0].get('event_time_absolute', False)
                if event_time_absolute:
                    start_col = 'start_absolute'
                    end_col = 'end_absolute'
                else:
                    start_col = 'start_time'
                    end_col = 'end_time'
                
                video_start = selected_animals[0]['video_start_fiber']
                
                for _, row in event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row[start_col] - video_start
                    end_time = row[end_col] - video_start
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Raw fiber data plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            self.set_status("Raw plot failed")
    
    def smooth_data(self):
        """Apply smoothing to selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            window_size = self.smooth_window.get()
            poly_order = self.smooth_order.get()
            
            if window_size % 2 == 0:
                window_size += 1
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                for channel_num in animal_data['active_channels']:
                    if channel_num in animal_data['channel_data']:
                        target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in animal_data['preprocessed_data'].columns:
                            smoothed_col = f"CH{channel_num}_{self.target_signal_var.get()}_smoothed"
                            animal_data['preprocessed_data'][smoothed_col] = savgol_filter(
                                animal_data['preprocessed_data'][target_col], window_size, poly_order)
            
            self.set_status(f"Data smoothed with window={window_size}, order={poly_order}")
            messagebox.showinfo("Success", f"Smoothing applied to {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed: {str(e)}")
            self.set_status("Smoothing failed")
    
    def baseline_correction(self):
        """Apply baseline correction to selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            model_type = self.baseline_model.get()
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
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
            
            self.set_status(f"Baseline correction applied ({model_type} model)")
            messagebox.showinfo("Success", f"Baseline correction applied to {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Baseline correction failed: {str(e)}")
            self.set_status("Baseline correction failed")
    
    def motion_correction(self):
        """Apply motion correction to selected animals"""
        if self.reference_signal_var.get() != "410":
            messagebox.showwarning("Invalid Reference", "Motion correction requires 410nm as reference signal")
            return
            
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if animal_data.get('preprocessed_data') is None:
                    animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                
                for channel_num in animal_data['active_channels']:
                    if channel_num in animal_data['channel_data']:
                        ref_col = animal_data['channel_data'][channel_num].get('410')
                        if not ref_col or ref_col not in animal_data['preprocessed_data'].columns:
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
            
            self.set_status("Motion correction applied")
            messagebox.showinfo("Success", f"Motion correction applied to {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Motion correction failed: {str(e)}")
            self.set_status("Motion correction failed")
    
    def apply_preprocessing(self):
        """Apply preprocessing to selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            if self.apply_smooth.get():
                self.smooth_data()
            if self.apply_baseline.get():
                self.baseline_correction()
            if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                self.motion_correction()
            
            messagebox.showinfo("Success", f"Preprocessing applied to {len(selected_indices)} animals")
            self.set_status(f"Preprocessing applied to {len(selected_indices)} animals")
                
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.set_status("Preprocessing failed")

    def calculate_and_plot_dff(self):
        """Calculate dF/F for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if animal_data.get('preprocessed_data') is None:
                    animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()

                animal_data['dff_data'] = {}
                
                time_col = animal_data['channels']['time']
                time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                
                baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                
                if not any(baseline_mask):
                    continue
                    
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
                                continue
                        elif ref_wavelength == "410" and not self.apply_baseline.get():
                            if f"CH{channel_num}_fitted_ref" in animal_data['preprocessed_data'].columns:
                                fitted_ref = animal_data['preprocessed_data'][f"CH{channel_num}_fitted_ref"]
                                dff_data = (raw_target - fitted_ref) / fitted_ref
                            else:
                                continue
                        elif ref_wavelength == "baseline" and self.apply_baseline.get():
                            if f"CH{channel_num}_baseline_pred" in animal_data['preprocessed_data'].columns:
                                baseline_fitted = animal_data['preprocessed_data'][f"CH{channel_num}_baseline_pred"]
                                baseline_median = np.median(raw_target[baseline_mask])
                                dff_data = (raw_target - baseline_fitted) / baseline_median
                            else:
                                continue
                        elif ref_wavelength == "baseline" and not self.apply_baseline.get():
                            baseline_median = np.median(raw_target[baseline_mask])
                            dff_data = (raw_target - baseline_median) / baseline_median
                        
                        dff_col = f"CH{channel_num}_dff"
                        animal_data['preprocessed_data'][dff_col] = dff_data
                        animal_data['dff_data'][channel_num] = dff_data
            
            self.set_status("ΔF/F calculated")
            messagebox.showinfo("Success", f"ΔF/F calculated for {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"ΔF/F calculation failed: {str(e)}")
            self.set_status("ΔF/F calculation failed")
    
    def calculate_and_plot_zscore(self):
        """Calculate z-score for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if not animal_data.get('dff_data'):
                    continue
                    
                time_col = animal_data['channels']['time']
                time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                
                baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                
                if not any(baseline_mask):
                    continue
                    
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
            messagebox.showinfo("Success", f"Z-score calculated for {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Z-score calculation failed: {str(e)}")
            self.set_status("Z-score calculation failed")
    
    def compute_event_zscore_from_dff(self, dff_data, time_data, event_start, event_end, pre_window, post_window, post_flag):
        """Compute event-related z-score from dF/F data"""
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

    def show_event_activity_dialog(self):
        """Show dialog for event activity and heatmap parameters"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Get available event types from selected animals
        event_types = set()
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'event_data' in animal_data:
                types = animal_data['event_data']['Event Type'].unique()
                event_types.update([t for t in types if t != 0])
        
        if not event_types:
            messagebox.showwarning("No Events", "No valid event types found")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Event-Activity & Heatmap Settings")
        dialog.geometry("400x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Spacing option
        ttk.Label(main_frame, text="Event Spacing:").grid(row=0, column=0, sticky="w", pady=5)
        spacing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Apply spacing", variable=spacing_var).grid(row=0, column=1, sticky="w")
        
        ttk.Label(main_frame, text="Spacing multiplier:").grid(row=1, column=0, sticky="w", pady=5)
        spacing_mult = ttk.Entry(main_frame, width=10)
        spacing_mult.insert(0, "0.6")
        spacing_mult.grid(row=1, column=1, sticky="w")
        
        # Event type selection
        ttk.Label(main_frame, text="Event Type:").grid(row=2, column=0, sticky="w", pady=5)
        event_type_var = tk.StringVar(value=str(min(event_types)))
        event_menu = ttk.OptionMenu(main_frame, event_type_var, str(min(event_types)), *[str(t) for t in sorted(event_types)])
        event_menu.grid(row=2, column=1, sticky="ew", pady=5)
        
        # Time windows
        ttk.Label(main_frame, text="Pre-event Window (s):").grid(row=3, column=0, sticky="w", pady=5)
        pre_entry = ttk.Entry(main_frame, width=10)
        pre_entry.insert(0, "10")
        pre_entry.grid(row=3, column=1, sticky="w")
        
        ttk.Label(main_frame, text="Post-event Window (s):").grid(row=4, column=0, sticky="w", pady=5)
        post_entry = ttk.Entry(main_frame, width=10)
        post_entry.insert(0, "20")
        post_entry.grid(row=4, column=1, sticky="w")
        
        # Smoothing options
        ttk.Label(main_frame, text="Smoothing Method:").grid(row=5, column=0, sticky="w", pady=5)
        smooth_var = tk.StringVar(value="None")
        smooth_menu = ttk.OptionMenu(main_frame, smooth_var, "None", "None", "Moving Average", "Savitzky-Golay")
        smooth_menu.grid(row=5, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="Smooth Window:").grid(row=6, column=0, sticky="w", pady=5)
        smooth_window = ttk.Entry(main_frame, width=10)
        smooth_window.insert(0, "11")
        smooth_window.grid(row=6, column=1, sticky="w")
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        def preview():
            try:
                params = {
                    'spacing': spacing_var.get(),
                    'spacing_mult': float(spacing_mult.get()),
                    'event_type': int(event_type_var.get()),
                    'pre_window': float(pre_entry.get()),
                    'post_window': float(post_entry.get()),
                    'smooth_method': smooth_var.get(),
                    'smooth_window': int(smooth_window.get()) if smooth_var.get() != "None" else None
                }
                self.plot_event_activity_and_heatmap(params, preview=True)
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        
        def apply():
            try:
                params = {
                    'spacing': spacing_var.get(),
                    'spacing_mult': float(spacing_mult.get()),
                    'event_type': int(event_type_var.get()),
                    'pre_window': float(pre_entry.get()),
                    'post_window': float(post_entry.get()),
                    'smooth_method': smooth_var.get(),
                    'smooth_window': int(smooth_window.get()) if smooth_var.get() != "None" else None
                }
                self.plot_event_activity_and_heatmap(params, preview=False)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        
        ttk.Button(btn_frame, text="Preview", command=preview).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Apply", command=apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=dialog.destroy).pack(side="left", padx=5)

    def plot_event_activity_and_heatmap(self, params, preview=False):
        """Plot event-related activity and two types of heatmaps"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        try:
            # Create figure with 3 subplots
            self.fig.clear()
            gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
            ax1 = self.fig.add_subplot(gs[0])  # Event-related activity
            ax2 = self.fig.add_subplot(gs[1])  # Heatmap: event-animal
            ax3 = self.fig.add_subplot(gs[2])  # Heatmap: animal-event
            
            # Collect data from selected animals
            all_data = []
            animal_labels = []
            event_labels = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                    continue
                
                events = animal_data['event_data'][animal_data['event_data']['Event Type'] == params['event_type']]
                events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]

                if events.empty:
                    continue
                
                time_col = animal_data['channels']['time']
                time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                
                for channel_num in animal_data['active_channels']:
                    if channel_num in animal_data['dff_data']:
                        dff_data = animal_data['dff_data'][channel_num]
                        
                        for event_idx, (_, event) in enumerate(events.iterrows()):
                            if animal_data.get('event_time_absolute', False):
                                start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                end_time = event['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start_time = event['start_time']
                                end_time = event['end_time']
                            
                            duration = end_time - start_time

                            event_time_rel, event_zscore, _ = self.compute_event_zscore_from_dff(
                                dff_data, time_data, start_time, end_time, 
                                params['pre_window'], params['post_window'], post_flag=True
                            )
                            
                            if len(event_time_rel) > 0:
                                # Apply smoothing if requested
                                if params['smooth_method'] == "Moving Average":
                                    kernel = np.ones(params['smooth_window']) / params['smooth_window']
                                    event_zscore = np.convolve(event_zscore, kernel, mode='same')
                                elif params['smooth_method'] == "Savitzky-Golay":
                                    event_zscore = savgol_filter(event_zscore, params['smooth_window'], 3)
                                
                                all_data.append({
                                    'animal_id': animal_data['animal_id'],
                                    'event_idx': event_idx,
                                    'time': event_time_rel,
                                    'zscore': event_zscore
                                })
                                animal_labels.append(f"{animal_data['animal_id']}")
                                event_labels.append(f"Event {event_idx + 1}")
            
            if not all_data:
                messagebox.showwarning("No Data", "No valid data to plot")
                return
            
            # Plot 1: Event-related activity with optional spacing
            if params['spacing']:
                all_zscores = [d['zscore'] for d in all_data]
                data_range = max([max(z) for z in all_zscores]) - min([min(z) for z in all_zscores])
                y_spacing = data_range * params['spacing_mult']
            else:
                y_spacing = 0
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
            for i, data_item in enumerate(all_data):
                y_offset = i * y_spacing
                ax1.plot(data_item['time'], data_item['zscore'] + y_offset, 
                        color=colors[i], label=f"{data_item['animal_id']} E{data_item['event_idx']+1}")
            
            ax1.axvline(0, color='#000000', linestyle='--', linewidth=1)
            ax1.axvline(duration, color='#000000', linestyle='--', linewidth=1)
            for _, event2 in events2.iterrows():
                if animal_data.get('event_time_absolute', False):
                    start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                    end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                else:
                    start2 = event2['start_time']
                    end2 = event2['end_time']
                
                for _, event1 in events.iterrows():
                    if animal_data.get('event_time_absolute', False):
                        event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                        event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                    else:
                        event1_start = event1['start_time']
                        event1_end = event1['end_time']
                    
                    if start2 >= event1_start and end2 <= event1_end:
                        rel_start = start2 - event1_start
                        ax1.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                label='Type 2 Event' if 'Type 2 Event' not in ax1.get_legend_handles_labels()[1] else "")
                            
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Z-Score")
            ax1.set_title("Event-Related Activity")
            ax1.grid(False)

            all_lines = ax1.get_lines()
            if all_lines:
                x_vals = np.concatenate([line.get_xdata() for line in all_lines])
                ax1.set_xlim(x_vals.min(), x_vals.max())
            
            t_min = min(d['time'].min() for d in all_data)
            t_max = max(d['time'].max() for d in all_data)
            n_pts = 200
            t_common = np.linspace(t_min, t_max, n_pts)

            # Plot 2: Heatmap event-animal (each event shows all animals)
            event_animal_matrix = []
            event_animal_labels = []
            
            unique_events = sorted(set(d['event_idx'] for d in all_data))
            for event_idx in unique_events:
                event_data_list = [d for d in all_data if d['event_idx'] == event_idx]
                for d in event_data_list:
                    z_interp = np.interp(t_common, d['time'], d['zscore'])
                    event_animal_matrix.append(z_interp)
                    event_animal_labels.append(f"E{event_idx+1} {d['animal_id']}")
                        
            if event_animal_matrix:
                colors_map = LinearSegmentedColormap.from_list("custom", ["blue", "white", "red"], N=256)
                im2 = ax2.imshow(np.array(event_animal_matrix), aspect='auto', cmap=colors_map,
                               extent=[all_data[0]['time'][0], all_data[0]['time'][-1], 0, len(event_animal_matrix)])
                ax2.set_ylabel("Event-Animal")
                ax2.set_title("Heatmap: Event-Animal")
                ax2.axvline(0, color='#000000', linestyle='--', linewidth=1)
                ax2.axvline(duration, color='#000000', linestyle='--', linewidth=1)

                for _, event2 in events2.iterrows():
                    if animal_data.get('event_time_absolute', False):
                        start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                        end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                    else:
                        start2 = event2['start_time']
                        end2 = event2['end_time']
                    
                    for _, event1 in events.iterrows():
                        if animal_data.get('event_time_absolute', False):
                            event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                            event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                        else:
                            event1_start = event1['start_time']
                            event1_end = event1['end_time']
                        
                        if start2 >= event1_start and end2 <= event1_end:
                            rel_start = start2 - event1_start
                            ax2.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                    label='Type 2 Event' if 'Type 2 Event' not in ax2.get_legend_handles_labels()[1] else "")
            
            # Plot 3: Heatmap animal-event (each animal shows all events)
            animal_event_matrix = []
            animal_event_labels = []
            
            unique_animals = sorted(set(d['animal_id'] for d in all_data))
            for animal_id in unique_animals:
                animal_data_list = [d for d in all_data if d['animal_id'] == animal_id]
                for d in animal_data_list:
                    z_interp = np.interp(t_common, d['time'], d['zscore'])
                    animal_event_matrix.append(z_interp)
                    animal_event_labels.append(f"{animal_id}")
            
            if animal_event_matrix:
                im3 = ax3.imshow(np.array(animal_event_matrix), aspect='auto', cmap=colors_map,
                               extent=[all_data[0]['time'][0], all_data[0]['time'][-1], 0, len(animal_event_matrix)])
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Animal-Event")
                ax3.set_title("Heatmap: Animal-Event")
                ax3.axvline(0, color='#000000', linestyle='--', linewidth=1)
                ax3.axvline(duration, color='#000000', linestyle='--', linewidth=1)

                for _, event2 in events2.iterrows():
                    if animal_data.get('event_time_absolute', False):
                        start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                        end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                    else:
                        start2 = event2['start_time']
                        end2 = event2['end_time']
                    
                    for _, event1 in events.iterrows():
                        if animal_data.get('event_time_absolute', False):
                            event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                            event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                        else:
                            event1_start = event1['start_time']
                            event1_end = event1['end_time']
                        
                        if start2 >= event1_start and end2 <= event1_end:
                            rel_start = start2 - event1_start
                            ax3.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                    label='Type 2 Event' if 'Type 2 Event' not in ax3.get_legend_handles_labels()[1] else "")
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Event activity and heatmaps plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")
            traceback.print_exc()

    def show_experiment_activity_dialog(self):
        """Show dialog for experiment activity parameters"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Get available event types
        event_types = set()
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'event_data' in animal_data:
                types = animal_data['event_data']['Event Type'].unique()
                event_types.update([t for t in types if t != 0])
        
        if not event_types:
            messagebox.showwarning("No Events", "No valid event types found")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Experiment-Activity Settings")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Event type selection
        ttk.Label(main_frame, text="Event Type:").grid(row=0, column=0, sticky="w", pady=5)
        event_type_var = tk.StringVar(value=str(min(event_types)))
        event_menu = ttk.OptionMenu(main_frame, event_type_var, str(min(event_types)), *[str(t) for t in sorted(event_types)])
        event_menu.grid(row=0, column=1, sticky="ew", pady=5)
        
        # Time windows
        ttk.Label(main_frame, text="Pre-event Window (s):").grid(row=1, column=0, sticky="w", pady=5)
        pre_entry = ttk.Entry(main_frame, width=10)
        pre_entry.insert(0, "10")
        pre_entry.grid(row=1, column=1, sticky="w")
        
        ttk.Label(main_frame, text="Post-event Window (s):").grid(row=2, column=0, sticky="w", pady=5)
        post_entry = ttk.Entry(main_frame, width=10)
        post_entry.insert(0, "20")
        post_entry.grid(row=2, column=1, sticky="w")
        
        # Smoothing options
        ttk.Label(main_frame, text="Smoothing Method:").grid(row=3, column=0, sticky="w", pady=5)
        smooth_var = tk.StringVar(value="None")
        smooth_menu = ttk.OptionMenu(main_frame, smooth_var, "None", "None", "Moving Average", "Savitzky-Golay")
        smooth_menu.grid(row=3, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="Smooth Window:").grid(row=4, column=0, sticky="w", pady=5)
        smooth_window = ttk.Entry(main_frame, width=10)
        smooth_window.insert(0, "11")
        smooth_window.grid(row=4, column=1, sticky="w")
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        def preview():
            try:
                params = {
                    'event_type': int(event_type_var.get()),
                    'pre_window': float(pre_entry.get()),
                    'post_window': float(post_entry.get()),
                    'smooth_method': smooth_var.get(),
                    'smooth_window': int(smooth_window.get()) if smooth_var.get() != "None" else None
                }
                self.plot_experiment_activity(params, preview=True)
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        
        def apply():
            try:
                params = {
                    'event_type': int(event_type_var.get()),
                    'pre_window': float(pre_entry.get()),
                    'post_window': float(post_entry.get()),
                    'smooth_method': smooth_var.get(),
                    'smooth_window': int(smooth_window.get()) if smooth_var.get() != "None" else None
                }
                self.plot_experiment_activity(params, preview=False)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        
        ttk.Button(btn_frame, text="Preview", command=preview).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Apply", command=apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=dialog.destroy).pack(side="left", padx=5)

    def plot_experiment_activity(self, params, preview=False):
        """Plot experiment-related activity (both dF/F and z-score)"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        try:
            self.fig.clear()
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)
            
            all_dff_responses = []
            all_zscore_responses = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                    continue
                
                events = animal_data['event_data'][animal_data['event_data']['Event Type'] == params['event_type']]
                events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]

                if events.empty:
                    continue
                
                time_col = animal_data['channels']['time']
                time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                
                for channel_num in animal_data['active_channels']:
                    if channel_num not in animal_data['dff_data']:
                        continue
                    
                    dff_data = animal_data['dff_data'][channel_num]
                    
                    for _, event in events.iterrows():
                        if animal_data.get('event_time_absolute', False):
                            start_time = event['start_absolute'] - animal_data['video_start_fiber']
                            end_time = event['end_absolute'] - animal_data['video_start_fiber']
                        else:
                            start_time = event['start_time']
                            end_time = event['end_time']
                        
                        duration = end_time - start_time
                        
                        event_time_rel, event_dff, _ = self.compute_event_zscore_from_dff(
                            dff_data, time_data, start_time, end_time,
                            params['pre_window'], params['post_window'], post_flag=True
                        )
                        
                        if len(event_time_rel) > 0:
                            # Apply smoothing if requested
                            if params['smooth_method'] == "Moving Average":
                                kernel = np.ones(params['smooth_window']) / params['smooth_window']
                                event_dff_smooth = np.convolve(event_dff, kernel, mode='same')
                            elif params['smooth_method'] == "Savitzky-Golay":
                                event_dff_smooth = savgol_filter(event_dff, params['smooth_window'], 3)
                            else:
                                event_dff_smooth = event_dff
                            
                            all_dff_responses.append((event_time_rel, event_dff_smooth))
                            
                            # Calculate z-score
                            baseline_mask = event_time_rel < 0
                            if np.sum(baseline_mask) >= 2:
                                baseline_mean = np.mean(event_dff_smooth[baseline_mask])
                                baseline_std = np.std(event_dff_smooth[baseline_mask])
                                if baseline_std > 0:
                                    event_zscore = (event_dff_smooth - baseline_mean) / baseline_std
                                    all_zscore_responses.append((event_time_rel, event_zscore))
            
            if not all_dff_responses:
                messagebox.showwarning("No Data", "No valid data to plot")
                return
            
            # Interpolate and average dF/F
            fps = fps_fiber
            max_time = max([t[-1] for t, _ in all_dff_responses])
            min_time = min([t[0] for t, _ in all_dff_responses])
            common_time = np.linspace(min_time, max_time, int((max_time - min_time) * fps))
            
            all_dff_interp = []
            for time_rel, dff in all_dff_responses:
                if len(time_rel) > 1:
                    interp_dff = np.interp(common_time, time_rel, dff)
                    all_dff_interp.append(interp_dff)
            
            mean_dff = np.nanmean(all_dff_interp, axis=0)
            sem_dff = np.nanstd(all_dff_interp, axis=0, ddof=1) / np.sqrt(len(all_dff_interp))
            
            ax1.plot(common_time, mean_dff, color="#80ff00", linestyle='-', label='Mean ΔF/F')
            ax1.fill_between(common_time, mean_dff - sem_dff, mean_dff + sem_dff, color="#80ff00", alpha=0.2, label='± SEM')
            ax1.axvline(0, color='#000000', linestyle='--', linewidth=1)
            ax1.axvline(duration, color='#000000', linestyle='--', linewidth=1, label='Event End')
            # ax1.axvspan(0, duration, color="#585252", alpha=0.2, label='Event Duration')
            
            for _, event2 in events2.iterrows():
                if animal_data.get('event_time_absolute', False):
                    start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                    end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                else:
                    start2 = event2['start_time']
                    end2 = event2['end_time']
                
                for _, event1 in events.iterrows():
                    if animal_data.get('event_time_absolute', False):
                        event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                        event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                    else:
                        event1_start = event1['start_time']
                        event1_end = event1['end_time']
                    
                    if start2 >= event1_start and end2 <= event1_end:
                        rel_start = start2 - event1_start
                        ax1.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                label='Type 2 Event' if 'Type 2 Event' not in ax1.get_legend_handles_labels()[1] else "")
                        
            ax1.set_ylabel("ΔF/F")
            ax1.set_title("Experiment-Related Activity (ΔF/F)")
            ax1.grid(False)
            ax1.legend()
            
            # Interpolate and average z-score
            all_zscore_interp = []
            for time_rel, zscore in all_zscore_responses:
                if len(time_rel) > 1:
                    interp_zscore = np.interp(common_time, time_rel, zscore)
                    all_zscore_interp.append(interp_zscore)
            
            mean_zscore = np.nanmean(all_zscore_interp, axis=0)
            sem_zscore = np.nanstd(all_zscore_interp, axis=0, ddof=1) / np.sqrt(len(all_zscore_interp))
            
            ax2.plot(common_time, mean_zscore, color="#80ff00", linestyle='-', label='Mean Z-score')
            ax2.fill_between(common_time, mean_zscore - sem_zscore, mean_zscore + sem_zscore, color="#80ff00", alpha=0.2, label='± SEM')
            ax2.axvline(0, color='#000000', linestyle='--', linewidth=1)
            ax2.axvline(duration, color='#000000', linestyle='--', linewidth=1, label='Event End')
            # ax2.axvspan(0, duration, color="#585252", alpha=0.2, label='Event Duration')
            
            for _, event2 in events2.iterrows():
                if animal_data.get('event_time_absolute', False):
                    start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                    end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                else:
                    start2 = event2['start_time']
                    end2 = event2['end_time']
                
                for _, event1 in events.iterrows():
                    if animal_data.get('event_time_absolute', False):
                        event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                        event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                    else:
                        event1_start = event1['start_time']
                        event1_end = event1['end_time']
                    
                    if start2 >= event1_start and end2 <= event1_end:
                        rel_start = start2 - event1_start
                        ax2.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                label='Type 2 Event' if 'Type 2 Event' not in ax2.get_legend_handles_labels()[1] else "")
                        
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Z-Score")
            ax2.set_title("Experiment-Related Activity (Z-Score)")
            ax2.grid(False)
            ax2.legend()
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Experiment activity plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")
            traceback.print_exc()

    def export_statistic_results(self):
        """Export statistics for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Check if any animal has event type 2
        has_type2 = False
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'event_data' in animal_data:
                if 2 in animal_data['event_data']['Event Type'].values:
                    has_type2 = True
                    break
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Statistics")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Time Windows (seconds):").grid(row=0, column=0, columnspan=2, sticky="w", pady=10)
        
        if has_type2:
            ttk.Label(main_frame, text="Pre Event Type 1 Start:").grid(row=1, column=0, sticky="w", pady=5)
            pre_start = ttk.Entry(main_frame, width=10)
            pre_start.insert(0, "5")
            pre_start.grid(row=1, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 Start:").grid(row=2, column=0, sticky="w", pady=5)
            post_start = ttk.Entry(main_frame, width=10)
            post_start.insert(0, "5")
            post_start.grid(row=2, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Pre Event Type 2 Start:").grid(row=3, column=0, sticky="w", pady=5)
            pre_type2 = ttk.Entry(main_frame, width=10)
            pre_type2.insert(0, "5")
            pre_type2.grid(row=3, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 End:").grid(row=4, column=0, sticky="w", pady=5)
            post_end = ttk.Entry(main_frame, width=10)
            post_end.insert(0, "5")
            post_end.grid(row=4, column=1, sticky="w")
        else:
            ttk.Label(main_frame, text="Pre Event Type 1 Start:").grid(row=1, column=0, sticky="w", pady=5)
            pre_start = ttk.Entry(main_frame, width=10)
            pre_start.insert(0, "5")
            pre_start.grid(row=1, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 Start:").grid(row=2, column=0, sticky="w", pady=5)
            post_start = ttk.Entry(main_frame, width=10)
            post_start.insert(0, "5")
            post_start.grid(row=2, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Pre Event Type 1 End:").grid(row=3, column=0, sticky="w", pady=5)
            pre_end = ttk.Entry(main_frame, width=10)
            pre_end.insert(0, "5")
            pre_end.grid(row=3, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 End:").grid(row=4, column=0, sticky="w", pady=5)
            post_end = ttk.Entry(main_frame, width=10)
            post_end.insert(0, "5")
            post_end.grid(row=4, column=1, sticky="w")
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        def export():
            try:
                all_stats = []
                
                for idx in selected_indices:
                    animal_data = self.multi_animal_data[idx]
                    if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    group = animal_data.get('group', 'Unknown')
                    
                    events_type1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 1]
                    if events_type1.empty:
                        continue
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num not in animal_data['dff_data']:
                            continue
                        
                        dff_data = animal_data['dff_data'][channel_num]
                        
                        for event_idx, (_, event) in enumerate(events_type1.iterrows()):
                            if animal_data.get('event_time_absolute', False):
                                start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                end_time = event['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start_time = event['start_time']
                                end_time = event['end_time']
                            
                            if has_type2:
                                # 4 windows + type 2 period
                                windows = [
                                    ('Pre Type1 Start', start_time - float(pre_start.get()), start_time),
                                    ('Post Type1 Start', start_time, start_time + float(post_start.get())),
                                    ('Pre Type2 Start', None, None),  # Will be filled if type2 exists
                                    ('Post Type1 End', end_time, end_time + float(post_end.get()))
                                ]
                                
                                # Find type 2 events during this type 1 event
                                events_type2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                                for _, event2 in events_type2.iterrows():
                                    if animal_data.get('event_time_absolute', False):
                                        type2_start = event2['start_absolute'] - animal_data['video_start_fiber']
                                        type2_end = event2['end_absolute'] - animal_data['video_start_fiber']
                                    else:
                                        type2_start = event2['start_time']
                                        type2_end = event2['end_time']
                                    
                                    if type2_start >= start_time and type2_end <= end_time:
                                        windows[2] = ('Pre Type2 Start', type2_start - float(pre_type2.get()), type2_start)
                                        windows.append(('Type2 Period', type2_start, type2_end))
                                        break
                            else:
                                # 4 windows
                                windows = [
                                    ('Pre Type1 Start', start_time - float(pre_start.get()), start_time),
                                    ('Post Type1 Start', start_time, start_time + float(post_start.get())),
                                    ('Pre Type1 End', end_time - float(pre_end.get()), end_time),
                                    ('Post Type1 End', end_time, end_time + float(post_end.get()))
                                ]
                            
                            for window_name, win_start, win_end in windows:
                                if win_start is None or win_end is None:
                                    continue
                                
                                mask = (time_data >= win_start) & (time_data <= win_end)
                                window_dff = dff_data[mask]
                                window_time = time_data[mask]
                                
                                if len(window_dff) > 0:
                                    mean_val = float(np.mean(window_dff))
                                    max_val = float(np.max(window_dff))
                                    min_val = float(np.min(window_dff))
                                    try:
                                        auc_val = float(np.trapezoid(window_dff, window_time))
                                    except AttributeError:
                                        auc_val = float(np.trapz(window_dff, window_time))
                                    
                                    all_stats.append({
                                        'Group': group,
                                        'Animal ID': animal_id,
                                        'Channel': channel_num,
                                        'Event Index': event_idx + 1,
                                        'Window': window_name,
                                        'Mean': mean_val,
                                        'Max': max_val,
                                        'Min': min_val,
                                        'AUC': auc_val
                                    })
                
                if not all_stats:
                    messagebox.showinfo("No Data", "No valid statistics calculated")
                    return
                
                df = pd.DataFrame(all_stats)
                
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    title="Save Statistics Results"
                )
                
                if save_path:
                    df.to_csv(save_path, index=False, float_format='%.10f')
                    messagebox.showinfo("Success", f"Statistics exported to:\n{save_path}")
                    self.set_status(f"Statistics exported: {len(all_stats)} records")
                    dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")
                traceback.print_exc()
        
        ttk.Button(btn_frame, text="Export", command=export).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

class FreezingAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Freezing Analyzer")
        self.root.state('zoomed')
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

        # Multi-animal list frame
        multi_animal_frame = ttk.LabelFrame(control_panel, text="Multi Animal Analysis")
        multi_animal_frame.pack(fill="x", padx=5, pady=5)
        
        list_frame = ttk.Frame(multi_animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(multi_animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Single Import", command=self.single_import).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Multi Import", command=self.multi_import).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        # Freezing analysis buttons
        freezing_frame = ttk.LabelFrame(control_panel, text="Freezing Analysis")
        freezing_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(freezing_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Plot Raster", command=self.plot_raster_with_event_selection).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Plot Freezing Timeline", command=self.plot_freezing_timeline).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
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
        
        ttk.Button(self.step2_frame, text="Apply Preprocessing", 
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
        
        ttk.Button(self.step3_frame, text="Calculate ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")

        if not hasattr(self, 'step4_frame') or not self.step4_frame:
            self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
            self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        self.step4_frame.update_idletasks()
        
        for widget in self.step4_frame.winfo_children():
            widget.destroy()
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text="Plot Event-Activity & Heatmap", 
                  command=self.show_event_activity_dialog).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Plot Experiment-Activity", 
                  command=self.show_experiment_activity_dialog).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Export Statistics", 
                  command=self.export_statistic_results).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

    def calculate_fps_from_fiber(self):
        """Calculate FPS from DLC frames and fiber video duration"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'dlc_data' in animal_data and 'video_start_fiber' in animal_data and 'video_end_fiber' in animal_data:
                total_frames = len(animal_data['dlc_data'])
                video_duration = animal_data['video_end_fiber'] - animal_data['video_start_fiber']
                if video_duration > 0:
                    animal_data['fps'] = total_frames / video_duration
                    print(f"FPS for {animal_data['animal_id']}: {animal_data['fps']:.2f}")
                    self.set_status(f"FPS for {animal_data['animal_id']}: {animal_data['fps']:.2f}")

    def compute_freezing_for_animal(self, animal_data):
        """Compute freezing analysis for a single animal"""
        try:
            dlc_result = animal_data['dlc_data']
            event_data = animal_data['event_data']
            fps = animal_data.get('fps', 30)
            
            scorer = dlc_result.columns.levels[0][0]
            left_ear = 'left_ear'
            right_ear = 'right_ear'
            tail_base = 'tail_base'

            le_x = dlc_result.loc[:, (scorer, left_ear, 'x')].values.astype(float)
            le_y = dlc_result.loc[:, (scorer, left_ear, 'y')].values.astype(float)
            le_r = dlc_result.loc[:, (scorer, left_ear, 'likelihood')].values.astype(float)
            
            re_x = dlc_result.loc[:, (scorer, right_ear, 'x')].values.astype(float)
            re_y = dlc_result.loc[:, (scorer, right_ear, 'y')].values.astype(float)
            re_r = dlc_result.loc[:, (scorer, right_ear, 'likelihood')].values.astype(float)
            
            tb_x = dlc_result.loc[:, (scorer, tail_base, 'x')].values.astype(float)
            tb_y = dlc_result.loc[:, (scorer, tail_base, 'y')].values.astype(float)
            tb_r = dlc_result.loc[:, (scorer, tail_base, 'likelihood')].values.astype(float)

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

            # Calculate freezing by event periods
            freezing_results = []
            for _, event in event_data.iterrows():
                event_type = event['Event Type']
                if event_type not in [0, 1]:
                    continue
                
                start_frame = int(event['start_time'] * fps)
                end_frame = int(event['end_time'] * fps)
                
                event_frames = np.arange(start_frame, min(end_frame, len(speed_smooth)))
                freeze_frames = np.intersect1d(event_frames, freeze_idx)
                
                total_time = len(event_frames) / fps
                freeze_time = len(freeze_frames) / fps
                freeze_percent = (len(freeze_frames) / len(event_frames) * 100) if len(event_frames) > 0 else 0
                
                freezing_results.append({
                    'event_type': event_type,
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'total_time': total_time,
                    'freeze_time': freeze_time,
                    'freeze_percent': freeze_percent
                })

            dlc_freeze_status = np.zeros(len(speed_smooth), dtype=int)
            dlc_freeze_status[freeze_idx] = 1

            return {
                'freezing_results': freezing_results,
                'raw_speed': raw_dist,
                'interp_speed': dist_interp,
                'smooth_speed': speed_smooth,
                'freeze_threshold': threshold,
                'freeze_idx': freeze_idx,
                'dlc_freeze_status': dlc_freeze_status
            }
            
        except Exception as e:
            self.set_status(f"Error computing freezing: {str(e)}")
            traceback.print_exc()
            return None

    def plot_raster_with_event_selection(self):
        """Plot raster with event selection"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)
        
        # Get available events
        all_events = []
        if selected_indices:
            animal_data = self.multi_animal_data[selected_indices[0]]
            if 'event_data' in animal_data:
                event_data = animal_data['event_data']
                for idx, row in event_data.iterrows():
                    event_type = row['Event Type']
                    label = "wait" if event_type == 0 else "sound"
                    all_events.append(f"{label}{math.ceil((idx+1)/2)}")
        
        if not all_events:
            messagebox.showwarning("No Events", "No events found")
            return
        
        # Create dialog for event selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Event Range")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Start Event:").grid(row=0, column=0, sticky="w", pady=5)
        start_var = tk.StringVar(value=all_events[0])
        start_menu = ttk.OptionMenu(main_frame, start_var, all_events[0], *all_events)
        start_menu.grid(row=0, column=1, sticky="ew", pady=5)
        
        ttk.Label(main_frame, text="End Event:").grid(row=1, column=0, sticky="w", pady=5)
        end_var = tk.StringVar(value=all_events[-1])
        end_menu = ttk.OptionMenu(main_frame, end_var, all_events[-1], *all_events)
        end_menu.grid(row=1, column=1, sticky="ew", pady=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        def plot():
            start_idx = all_events.index(start_var.get())
            end_idx = all_events.index(end_var.get())
            dialog.destroy()
            self.plot_raster(selected_indices, start_idx, end_idx)
        
        ttk.Button(btn_frame, text="Plot", command=plot).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

    def plot_raster(self, selected_indices, start_event_idx, end_event_idx):
        """Plot freezing raster for selected event range"""
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            y_pos = 0
            y_ticks = []
            y_labels = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'dlc_freeze_status' not in animal_data:
                    continue
                
                animal_id = animal_data['animal_id']
                fps = animal_data.get('fps', 30)
                
                y_ticks.append(y_pos)
                y_labels.append(animal_id)
                
                dlc_freeze_status = animal_data['dlc_freeze_status']
                time_values = np.arange(len(dlc_freeze_status)) / fps
                
                # Plot events in range
                event_data = animal_data['event_data']
                for ev_idx in range(start_event_idx, end_event_idx + 1):
                    if ev_idx < len(event_data):
                        event = event_data.iloc[ev_idx]
                        event_type = event['Event Type']
                        color = "#ffffff" if event_type == 0 else "#0400fc"
                        
                        ax.add_patch(plt.Rectangle(
                            (event['start_time'], y_pos - 0.4),
                            event['end_time'] - event['start_time'],
                            0.8, color=color, alpha=0.5
                        ))
                
                # Plot freezing
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
                ax.set_title("Freezing Raster Plot")
                ax.grid(False)
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Raster plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raster: {str(e)}")
            traceback.print_exc()

    def plot_freezing_timeline(self):
        """Plot freezing timeline by event"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)
        
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Aggregate freezing by event
            all_event_labels = []
            all_freeze_percents = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'freezing_results' not in animal_data:
                    continue
                
                for i, result in enumerate(animal_data['freezing_results']):
                    event_type = result['event_type']
                    label = f"wait{math.ceil((i+1)/2)}" if event_type == 0 else f"sound{math.ceil((i+1)/2)}"
                    if label not in all_event_labels:
                        all_event_labels.append(label)
                    
                    if len(all_freeze_percents) <= len(all_event_labels) - 1:
                        all_freeze_percents.append([])
                    all_freeze_percents[all_event_labels.index(label)].append(result['freeze_percent'])
            
            # Calculate mean and SEM
            mean_percents = [np.mean(p) if p else 0 for p in all_freeze_percents]
            sem_percents = [np.std(p, ddof=1) / np.sqrt(len(p)) if len(p) > 1 else 0 for p in all_freeze_percents]
            
            x_pos = np.arange(len(all_event_labels))
            ax.bar(x_pos, mean_percents, yerr=sem_percents, capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_event_labels, rotation=45, ha='right')
            ax.set_ylabel("Freezing %")
            ax.set_title("Freezing Timeline by Event")
            ax.grid(False)
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Freezing timeline plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot timeline: {str(e)}")
            traceback.print_exc()

    def export_freezing_csv(self):
        """Export freezing results to CSV"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)
        
        try:
            all_data = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'freezing_results' not in animal_data:
                    continue
                
                animal_id = animal_data['animal_id']
                group = animal_data['group']
                
                for i, result in enumerate(animal_data['freezing_results']):
                    event_type = result['event_type']
                    label = f"wait{math.ceil((i+1)/2)}" if event_type == 0 else f"sound{math.ceil((i+1)/2)}"
                    
                    all_data.append({
                        'Group': group,
                        'Animal ID': animal_id,
                        'Event': label,
                        'Event Type': event_type,
                        'Start Time (s)': result['start_time'],
                        'End Time (s)': result['end_time'],
                        'Total Time (s)': result['total_time'],
                        'Freeze Time (s)': result['freeze_time'],
                        'Freeze %': result['freeze_percent']
                    })
            
            if not all_data:
                messagebox.showinfo("No Data", "No freezing data to export")
                return
            
            df = pd.DataFrame(all_data)
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Freezing Results"
            )
            
            if save_path:
                df.to_csv(save_path, index=False, float_format='%.10f')
                messagebox.showinfo("Success", f"Freezing results exported to:\n{save_path}")
                self.set_status(f"Freezing exported: {len(all_data)} records")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
            traceback.print_exc()

    def plot_compare_raw_cleaning(self):
        """Plot comparison of raw vs cleaned speed data"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices or len(selected_indices) != 1:
            messagebox.showwarning("Selection Error", "Please select exactly one animal")
            return
        
        animal_data = self.multi_animal_data[selected_indices[0]]
        
        # Calculate FPS and compute freezing if not done
        self.calculate_fps_from_fiber()
        if 'freezing_results' not in animal_data:
            result = self.compute_freezing_for_animal(animal_data)
            if result:
                animal_data.update(result)
        
        if 'raw_speed' not in animal_data:
            messagebox.showwarning("No Data", "No speed data available")
            return
        
        try:
            fps = animal_data.get('fps', 30)
            x_time = np.arange(len(animal_data['raw_speed'])) / fps
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Plot event backgrounds
            if 'event_data' in animal_data:
                for _, event in animal_data['event_data'].iterrows():
                    color = "#b6b6b6" if event['Event Type'] == 0 else "#fcb500"
                    ax.axvspan(event['start_time'], event['end_time'], color=color, alpha=0.3)
            
            # Plot speed traces
            ax.plot(x_time, animal_data['raw_speed'], alpha=0.6, label='Raw Speed')
            ax.plot(x_time, animal_data['interp_speed'], alpha=0.6, label='Interpolated')
            ax.plot(x_time, animal_data['smooth_speed'], linewidth=1.5, label='Smoothed')
            ax.axhline(animal_data['freeze_threshold'], linestyle='--', color="#ff0000ff", label='Threshold')
            
            # Plot freezing zones
            in_freeze = np.zeros_like(animal_data['smooth_speed'])
            in_freeze[animal_data['freeze_idx']] = 1
            ax.fill_between(x_time, 0, max(animal_data['smooth_speed']), 
                           where=in_freeze > 0, color="#ff0000ff", alpha=0.3, label='Freezing')
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (px/frame)")
            ax.set_title(f"Speed Trace - {animal_data['animal_id']}")
            ax.legend()
            ax.grid(False)
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Raw vs cleaned plot complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")
            traceback.print_exc()

    def plot_freezing_with_fiber(self):
        """Plot freezing with fiber photometry data"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)
        
        try:
            self.fig.clear()
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = self.fig.add_subplot(gs[0])
            ax2 = self.fig.add_subplot(gs[1])
            
            # Plot fiber data
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
            for idx_pos, idx in enumerate(selected_indices):
                animal_data = self.multi_animal_data[idx]
                if 'fiber_cropped' not in animal_data:
                    continue
                
                time_col = animal_data['channels']['time']
                time_data = animal_data['fiber_cropped'][time_col] - animal_data['video_start_fiber']
                
                for channel_num in animal_data['active_channels']:
                    if channel_num in animal_data['channel_data']:
                        target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in animal_data['fiber_cropped'].columns:
                            ax1.plot(time_data, animal_data['fiber_cropped'][target_col],
                                   color=colors[idx_pos], linewidth=1, 
                                   label=f'{animal_data["animal_id"]} CH{channel_num}')
            
            ax1.set_title('Freezing with Fiber Photometry')
            ax1.set_ylabel("Fiber Signal")
            ax1.grid(False)
            ax1.legend()
            
            # Plot freezing raster
            y_ticks = []
            y_labels = []
            
            for idx_pos, idx in enumerate(selected_indices):
                animal_data = self.multi_animal_data[idx]
                if 'dlc_freeze_status' not in animal_data:
                    continue
                
                fps = animal_data.get('fps', 30)
                dlc_freeze_status = animal_data['dlc_freeze_status']
                time_values = np.arange(len(dlc_freeze_status)) / fps
                freezing_times = time_values[dlc_freeze_status == 1]
                
                y_pos = idx_pos
                y_ticks.append(y_pos)
                y_labels.append(animal_data['animal_id'])
                
                for t in freezing_times:
                    ax2.vlines(t, y_pos - 0.4, y_pos + 0.4, color='black', 
                              linewidth=0.5, alpha=0.7)
            
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels(y_labels)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Animals")
            ax2.grid(False)
            
            # Add event backgrounds
            if selected_indices and 'event_data' in self.multi_animal_data[selected_indices[0]]:
                event_data = self.multi_animal_data[selected_indices[0]]['event_data']
                video_start = self.multi_animal_data[selected_indices[0]]['video_start_fiber']
                
                for _, row in event_data.iterrows():
                    color = "#b6b6b6" if row['Event Type'] == 0 else "#fcb500"
                    start_time = row['start_time']
                    end_time = row['end_time']
                    
                    ax1.axvspan(start_time, end_time, color=color, alpha=0.3)
                    ax2.axvspan(start_time, end_time, color=color, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.set_status("Freezing with fiber plotted")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")
            traceback.print_exc()


class PupilAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Pupil Analyzer")
        self.root.state('zoomed')
        self.dlc_results_path = ""
        self.timestamp_path = ""
        self.ast2_path = ""
        self.pupil_data = {}
        self.aligned_data = {}
        self.cam_id = None
        self.experiment_start = None
        self.selected_files = []
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)
        
        # Multi-animal list frame
        multi_animal_frame = ttk.LabelFrame(control_panel, text="Multi Animal Analysis")
        multi_animal_frame.pack(fill="x", padx=5, pady=5)
        
        list_frame = ttk.Frame(multi_animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(multi_animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Single Import", command=self.single_import).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Multi Import", command=self.multi_import).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        # Pupil analysis buttons
        pupil_frame = ttk.LabelFrame(control_panel, text="Pupil Analysis")
        pupil_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(pupil_frame)
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
        
        ttk.Button(self.step2_frame, text="Apply Preprocessing", 
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
        
        ttk.Button(self.step3_frame, text="Calculate ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")

        if not hasattr(self, 'step4_frame') or not self.step4_frame:
            self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
            self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        self.step4_frame.update_idletasks()
        
        for widget in self.step4_frame.winfo_children():
            widget.destroy()
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text="Plot Event-Activity & Heatmap", 
                  command=self.show_event_activity_dialog).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Plot Experiment-Activity", 
                  command=self.show_experiment_activity_dialog).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Export Statistics", 
                  command=self.export_statistic_results).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

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

    def plot_pupil_distance(self):
        """Plot pupil distance for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
        
        for idx_pos, idx in enumerate(selected_indices):
            animal_data = self.multi_animal_data[idx]
            if 'pupil_data' not in animal_data:
                # Run pupil analysis if not done
                self.run_pupil_analysis_for_animal(animal_data)
            
            if 'pupil_data' in animal_data:
                ax.plot(animal_data['pupil_data']['time'], 
                       animal_data['pupil_data']['distance'],
                       color=colors[idx_pos], linewidth=1.5,
                       label=animal_data['animal_id'])
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil Distance (pixels)")
        ax.set_title("Pupil Distance")
        ax.grid(False)
        ax.legend()
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.set_status("Pupil distance plotted")

    def run_pupil_analysis_for_animal(self, animal_data):
        """Run pupil analysis for a single animal"""
        try:
            if 'dlc_data' not in animal_data or 'experiment_start' not in animal_data:
                return
            
            rawdata = animal_data['dlc_data']
            
            bodyparts = rawdata.iloc[0, 1::3].values
            coordinates = {}
            for i in range(len(bodyparts)):
                tmp = {}
                tmp['x'] = rawdata.iloc[2:, 3*i+1].values.astype(float)
                tmp['y'] = -rawdata.iloc[2:, 3*i+2].values.astype(float)
                tmp['reliability'] = rawdata.iloc[2:, 3*i+3].values.astype(float)
                coordinates[bodyparts[i]] = tmp
            
            reliability_thresh = 0.99
            rlb_idx = {}
            for part in ['pupil_top', 'pupil_bottom']:
                rlb_idx[part] = np.where(coordinates[part]['reliability'] >= reliability_thresh)[0]
            
            common_idx = np.intersect1d(rlb_idx['pupil_top'], rlb_idx['pupil_bottom'])
            
            fps = 90
            relative_time = np.arange(0, len(coordinates['pupil_top']['x'])) / fps
            absolute_time = animal_data['experiment_start'] + relative_time
            time_span_r = absolute_time[common_idx]
            
            pupil_top_x_r = coordinates['pupil_top']['x'][common_idx]
            pupil_top_y_r = coordinates['pupil_top']['y'][common_idx]
            pupil_bottom_x_r = coordinates['pupil_bottom']['x'][common_idx]
            pupil_bottom_y_r = coordinates['pupil_bottom']['y'][common_idx]
            
            pupil_dist = np.sqrt((pupil_top_x_r - pupil_bottom_x_r)**2 + 
                                (pupil_top_y_r - pupil_bottom_y_r)**2)
            
            animal_data['pupil_data'] = {
                'time': time_span_r,
                'distance': pupil_dist,
                'coordinates': coordinates,
                'common_idx': common_idx
            }
            
        except Exception as e:
            self.set_status(f"Pupil analysis failed for {animal_data['animal_id']}: {str(e)}")

    def plot_combined_data(self):
        """Plot combined pupil, speed, and fiber data"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        self.fig.clear()
        ax1 = self.fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
        
        for idx_pos, idx in enumerate(selected_indices):
            animal_data = self.multi_animal_data[idx]
            
            if 'aligned_data' in animal_data:
                ax1.plot(animal_data['aligned_data']['time'],
                        animal_data['aligned_data']['speed'],
                        color=colors[idx_pos], linewidth=1.5,
                        label=f"{animal_data['animal_id']} Speed")
            
            if 'pupil_data' in animal_data:
                ax2.plot(animal_data['pupil_data']['time'],
                        animal_data['pupil_data']['distance'],
                        color=colors[idx_pos], linestyle='--', linewidth=1.5,
                        label=f"{animal_data['animal_id']} Pupil")
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Speed", color='blue')
        ax2.set_ylabel("Pupil Distance", color='red')
        ax1.set_title("Combined Pupil and Speed Data")
        ax1.grid(False)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.set_status("Combined data plotted")

    def plot_pupil_with_fiber(self):
        """Plot pupil data with fiber photometry"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        self.fig.clear()
        ax1 = self.fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
        
        for idx_pos, idx in enumerate(selected_indices):
            animal_data = self.multi_animal_data[idx]
            
            if 'pupil_data' in animal_data:
                ax1.plot(animal_data['pupil_data']['time'],
                        animal_data['pupil_data']['distance'],
                        color=colors[idx_pos], linewidth=1.5,
                        label=f"{animal_data['animal_id']} Pupil")
            
            if 'fiber_cropped' in animal_data:
                time_col = animal_data['channels']['time']
                time_data = animal_data['fiber_cropped'][time_col]
                
                for channel_num in animal_data['active_channels']:
                    if channel_num in animal_data['channel_data']:
                        target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                        if target_col and target_col in animal_data['fiber_cropped'].columns:
                            ax2.plot(time_data, animal_data['fiber_cropped'][target_col],
                                   color=colors[idx_pos], linestyle='--', linewidth=1.5,
                                   label=f"{animal_data['animal_id']} Fiber")
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pupil Distance", color='red')
        ax2.set_ylabel("Fiber Signal", color='blue')
        ax1.set_title("Pupil with Fiber Photometry")
        ax1.grid(False)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.set_status("Pupil with fiber plotted")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModeSelectionApp(root)
    root.mainloop()