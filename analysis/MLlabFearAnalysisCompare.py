# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 09:31:02 2025

Edit on Sun Apr 20 21:39:30 2025
Rename: MLlabFearAnalysis

@author: Pulee
"""

import os
import tkinter as tk
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from itertools import groupby
from operator import itemgetter
from tkinter import ttk, filedialog, messagebox, simpledialog
from scipy.interpolate import interp1d
from collections import defaultdict
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread")

class FreezingAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Conditioning Freezing Analyzer")
        self.root.geometry("450x450")
        self.root.resizable(False, False)

        self.dlc_results_path = ""
        self.eztrack_path = ""
        self.timestamp_path = ""
        self.video_path = ""
        self.freezing_data = {}
        self.eztrack_result = None
        self.event_data = None
        self.window_freeze = None
        self.window_freeze_eztrack = None
        self.start_frame = 0
        self.end_frame = 0
        self.method = ""
        self.fps = 30
        
        self.batch_folder = ""
        self.batch_data = []
        self.progress_window = None
        self.progress_var = None
        self.progress_label = None
        self.cancel_batch = False

        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky="NSEW")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        ttk.Button(main_frame, text="Load DLC Results", 
                  command=self.load_DLC_Results).grid(row=0, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Load EZTrack CSV", 
                  command=self.load_EZTrack_CSV).grid(row=0, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Load Timestamp CSV", 
                  command=self.load_timestamps).grid(row=1, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Load Original Video", 
                  command=self.load_video).grid(row=1, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Run Freezing Analysis (DLC)", 
                  command=self.analyze_dlc).grid(row=2, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Run Freezing Analysis (EZTrack)", 
                  command=self.analyze_eztrack).grid(row=2, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Export Results to CSV (DLC)", 
                  command=self.export_csv_dlc).grid(row=3, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Export Results to CSV (EZTrack)", 
                  command=self.export_csv_eztrack).grid(row=3, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Plot Freezing Timeline (DLC)", 
                  command=self.plot_freezing_windows_dlc).grid(row=4, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Plot Freezing Timeline (EZTrack)", 
                  command=self.plot_freezing_windows_eztrack).grid(row=4, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Compare Raw vs Cleaned (DLC)", 
                  command=self.plot_compare_raw_cleaning).grid(row=5, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Compare DLC vs EZTrack", 
                  command=self.plot_compare_freezing).grid(row=5, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Show Video (DLC)", 
                  command=self.show_video_dlc).grid(row=6, column=0, pady=6, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Show Video (EZTrack)", 
                  command=self.show_video_eztrack).grid(row=6, column=1, pady=6, padx=5, sticky="EW")
        
        ttk.Button(main_frame, text="Add Batch Folder", 
                  command=self.load_batch_folder).grid(row=7, column=0, pady=10, padx=5, sticky="EW")
        ttk.Button(main_frame, text="Run Batch Analysis", 
                  command=self.run_batch_analysis).grid(row=7, column=1, pady=10, padx=5, sticky="EW")

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="EW")
    
    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def load_batch_folder(self):
        self.batch_folder = filedialog.askdirectory(title="Select Behavioural Folder")
        if not self.batch_folder:
            return
            
        self.set_status(f"Scanning folder: {self.batch_folder}")
        
        self.batch_data = []
        self.batch_results = defaultdict(lambda: defaultdict(dict))
        
        for root, dirs, files in os.walk(self.batch_folder):
            if any(f.endswith('.csv') for f in files):
                parts = root.split(os.sep)
                if len(parts) < 4:
                    continue
                
                exp_type = parts[-2]  # CFC, CFT, EXT, RET
                animal_id = parts[-1]  # 0066, 0112, etc.
                date = parts[-3]  # 20250513, etc.
                
                dlc_file = None
                eztrack_file = None
                timestamp_file = None
                
                for f in files:
                    if f.endswith('.csv'):
                        if 'DLC' in f and not f.endswith('FreezingOutput.csv'):
                            dlc_file = os.path.join(root, f)
                        elif 'FreezingOutput' in f:
                            eztrack_file = os.path.join(root, f)
                        elif 'events' in f:
                            timestamp_file = os.path.join(root, f)
                
                if dlc_file and eztrack_file and timestamp_file:
                    self.batch_data.append({
                        'path': root,
                        'exp_type': exp_type,
                        'animal_id': animal_id,
                        'date': date,
                        'dlc_file': dlc_file,
                        'eztrack_file': eztrack_file,
                        'timestamp_file': timestamp_file
                    })
        
        if not self.batch_data:
            messagebox.showwarning("No Data", "No valid data folders found in the selected directory.")
            self.set_status("No batch data found")
            return
            
        messagebox.showinfo("Batch Data Loaded", 
                           f"Found {len(self.batch_data)} datasets for batch processing.")
        self.set_status(f"Loaded {len(self.batch_data)} datasets for batch processing")
    
    def run_batch_analysis(self):
        if not self.batch_data:
            messagebox.showwarning("No Data", "Please load a batch folder first.")
            return
            
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Batch Processing")
        self.progress_window.geometry("400x150")
        self.progress_window.resizable(False, False)
        self.progress_window.protocol("WM_DELETE_WINDOW", self.cancel_batch_processing)
        
        ttk.Label(self.progress_window, text="Processing batch data...", font=("Arial", 10)).pack(pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)
        progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, maximum=len(self.batch_data))
        progress_bar.pack(fill='x', padx=20, pady=5)
        
        self.progress_label = ttk.Label(self.progress_window, text="0/0")
        self.progress_label.pack(pady=5)
        
        cancel_btn = ttk.Button(self.progress_window, text="Cancel", command=self.cancel_batch_processing)
        cancel_btn.pack(pady=10)
    
        self.cancel_batch = False
        threading.Thread(target=self._process_batch, daemon=True).start()
    
    def cancel_batch_processing(self):
        self.cancel_batch = True
        if self.progress_window:
            self.progress_window.destroy()
        self.set_status("Batch processing canceled")
    
    def _process_batch(self):
        total = len(self.batch_data)
        
        for idx, dataset in enumerate(self.batch_data):
            if self.cancel_batch:
                break
                
            self.progress_var.set(idx + 1)
            self.progress_label.config(text=f"Processing {idx+1}/{total}: {dataset['animal_id']}")
            self.progress_window.update()
            
            try:
                animal_id = dataset['animal_id']
                exp_type = dataset['exp_type']
                
                self.dlc_results_path = dataset['dlc_file']
                self.eztrack_path = dataset['eztrack_file']
                self.timestamp_path = dataset['timestamp_file']
                
                self.eztrack_result = None
                self.window_freeze = None
                self.window_freeze_eztrack = None
                self.dlc_freeze_status = None
                self.eztrack_freeze_status = None
                
                self.event_data = pd.read_csv(self.timestamp_path)
                
                video_files = [f for f in os.listdir(dataset['path']) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if video_files:
                    video_path = os.path.join(dataset['path'], video_files[0])
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        self.fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                
                self.analyze_dlc(show_message=False)
                dlc_result = self.window_freeze.copy() if self.window_freeze is not None else None
                
                self.eztrack_result = None
                self.analyze_eztrack(show_message=False)
                eztrack_result = self.window_freeze_eztrack.copy() if self.window_freeze_eztrack is not None else None
                
                if dlc_result is not None:
                    dlc_result.to_csv(os.path.join(dataset['path'], f"{animal_id}_D_result.csv"), index=False)
                if eztrack_result is not None:
                    eztrack_result.to_csv(os.path.join(dataset['path'], f"{animal_id}_E_result.csv"), index=False)
                
                if hasattr(self, 'dlc_freeze_status') and hasattr(self, 'eztrack_freeze_status'):
                    self.save_raster_plot(dataset['path'], animal_id, exp_type)
                
                if dlc_result is not None and eztrack_result is not None:
                    plt.figure(figsize=(12, 6))
                    
                    for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
                        state_rows = self.event_data[self.event_data['Event Type'] == state_code]
                        for _, row in state_rows.iterrows():
                            plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)
                    
                    plt.plot(dlc_result['time'], dlc_result['freeze_sec'], 'o-', color='black', label='DLC Freezing')
                    plt.plot(eztrack_result['time'], eztrack_result['freeze_sec'], 's-', color='blue', label='EZTrack Freezing')
                    
                    plt.xlabel("Time (s)")
                    plt.ylabel("Freezing Duration (s) per 30s window")
                    plt.title(f"Freezing Comparison: {animal_id} ({exp_type})")
                    plt.legend(loc='upper left')
                    
                    all_x = np.concatenate([dlc_result['time'].values, eztrack_result['time'].values])
                    min_x = np.floor(min(all_x) / 30) * 30
                    max_x = np.ceil(max(all_x) / 30) * 30
                    ticks = np.arange(min_x, max_x+1, 30)
                    plt.xticks(ticks)
                    plt.xlim(min_x, max_x)
                    plt.grid(False)
                    
                    plt.savefig(os.path.join(dataset['path'], f"{animal_id}_compare.png"))
                    plt.close()
                
                self.set_status(f"Processed {animal_id} ({exp_type})")
                
            except Exception as e:
                error_msg = f"Error processing {dataset['animal_id']}: {str(e)}"
                self.set_status(error_msg)
                print(traceback.format_exc())
        
        if not self.cancel_batch:
            self.save_summary_tables()
        
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
            
        if not self.cancel_batch:
            messagebox.showinfo("Batch Complete", "Batch processing completed successfully!")
            self.set_status("Batch processing completed")
    
    def save_raster_plot(self, save_path, animal_id, exp_type):
        min_len = min(len(self.dlc_freeze_status), len(self.eztrack_freeze_status))
        
        time_seconds = np.arange(min_len) / self.fps
        
        dlc_freezing = self.dlc_freeze_status[:min_len]
        eztrack_freezing = self.eztrack_freeze_status[:min_len]
        
        plt.figure(figsize=(15, 4))
        
        plt.eventplot(time_seconds[dlc_freezing == 1], 
                      orientation='horizontal', 
                      lineoffsets=1, 
                      linelengths=0.8, 
                      colors='red', 
                      label='DLC Freezing')
        
        plt.eventplot(time_seconds[eztrack_freezing == 1], 
                      orientation='horizontal', 
                      lineoffsets=2, 
                      linelengths=0.8, 
                      colors='blue', 
                      label='EZTrack Freezing')
        
        plt.yticks([1, 2], ['DLC', 'EZTrack'])
        plt.ylim(0.5, 2.5)
        
        plt.xlabel("Time (s)")
        plt.title(f"Freezing Raster Plot: {animal_id} ({exp_type})")
        # plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{animal_id}_raster.png"))
        plt.close()
    
    def save_summary_tables(self):
        summary_path = os.path.join(self.batch_folder, "Summary_Tables")
        os.makedirs(summary_path, exist_ok=True)
        
        summary_data = {
            'DLC': defaultdict(dict),
            'EZTrack': defaultdict(dict)
        }
        
        max_windows = defaultdict(int)
        
        for dataset in self.batch_data:
            animal_id = dataset['animal_id']
            exp_type = dataset['exp_type']
            
            dlc_file = os.path.join(dataset['path'], f"{animal_id}_D_result.csv")
            if os.path.exists(dlc_file):
                dlc_df = pd.read_csv(dlc_file)
                freeze_sec = dlc_df['freeze_sec'].tolist()
                summary_data['DLC'][exp_type][animal_id] = freeze_sec
                max_windows[exp_type] = max(max_windows[exp_type], len(freeze_sec))
            
            eztrack_file = os.path.join(dataset['path'], f"{animal_id}_E_result.csv")
            if os.path.exists(eztrack_file):
                eztrack_df = pd.read_csv(eztrack_file)
                freeze_sec = eztrack_df['freeze_sec'].tolist()
                summary_data['EZTrack'][exp_type][animal_id] = freeze_sec
                max_windows[exp_type] = max(max_windows[exp_type], len(freeze_sec))
        
        for method in ['DLC', 'EZTrack']:
            for exp_type in max_windows.keys():
                if exp_type not in summary_data[method]:
                    continue
                    
                columns = [f'Window_{i+1}' for i in range(max_windows[exp_type])]
                df = pd.DataFrame(columns=columns)
                
                for animal_id, freeze_data in summary_data[method][exp_type].items():
                    padded_data = freeze_data + [np.nan] * (max_windows[exp_type] - len(freeze_data))
                    df.loc[animal_id] = padded_data
                
                if not df.empty:
                    df.loc['Average'] = df.mean()
                
                filename = f"{method}_{exp_type}_summary.csv"
                df.to_csv(os.path.join(summary_path, filename), index=True)
        
        self.set_status(f"Summary tables saved to {summary_path}")
        messagebox.showinfo("Summary Complete", f"Summary tables saved to:\n{summary_path}")

    def analyze_dlc(self, show_message=True):
        if not all([self.dlc_results_path, self.timestamp_path]):
            if show_message:
                messagebox.showwarning("Missing Inputs", "Please load DLC results and timestamps first.")
            return

        self.set_status("Loading DLC result...")
        try:
            self.dlc_result = pd.read_csv(self.dlc_results_path, header=[0, 1, 2])
            self.set_status("Computing freezing data...")
            self.freezing_data = self.compute_freezing(self.event_data)
            if show_message:
                messagebox.showinfo("Done", "DLC Freezing analysis completed!")
            self.set_status("DLC analysis completed.")
        except Exception as e:
            if show_message:
                messagebox.showerror("Analysis Error", f"Failed to analyze DLC data: {str(e)}")
            self.set_status("DLC analysis failed.")
            print(traceback.format_exc())

    def analyze_eztrack(self, show_message=True):
        if not all([self.eztrack_path, self.timestamp_path]):
            if show_message:
                messagebox.showwarning("Missing Inputs", "Please load EZTrack CSV and timestamps first.")
            return

        try:
            self.eztrack_result = pd.read_csv(self.eztrack_path)
        except Exception as e:
            if show_message:
                messagebox.showwarning("No EZTrack Data", f"Failed to load EZTrack CSV: {str(e)}")
            return

        self.set_status("Computing EZTrack freezing data...")
        try:
            self.freezing_data_eztrack = self.compute_freezing_from_eztrack(self.fps)
            self.compute_freezing_by_window(30, self.fps)
            if show_message:
                messagebox.showinfo("Done", "EZTrack Freezing analysis completed!")
            self.set_status("EZTrack analysis completed.")
        except Exception as e:
            if show_message:
                messagebox.showerror("Analysis Error", f"Failed to analyze EZTrack data: {str(e)}")
            self.set_status("EZTrack analysis failed.")
            print(traceback.format_exc())

    def load_DLC_Results(self):
        self.dlc_results_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.dlc_results_path:
            messagebox.showinfo("DLC results loaded", f"Selected model:\n{self.dlc_results_path}")
            self.set_status("DLC results loaded.")

    def load_EZTrack_CSV(self):
        self.eztrack_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.eztrack_path:
            self.set_status("Loading EZTrack data...")
            try:
                self.eztrack_result = pd.read_csv(self.eztrack_path)
                messagebox.showinfo("EZTrack loaded", f"EZTrack data loaded successfully!")
                self.set_status("EZTrack data loaded.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load EZTrack CSV: {str(e)}")
                self.set_status("EZTrack load failed.")

    def load_timestamps(self):
        self.timestamp_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.timestamp_path:
            try:
                self.event_data = pd.read_csv(self.timestamp_path)
                messagebox.showinfo("Timestamps Loaded", f"Selected timestamp file:\n{self.timestamp_path}")
                self.set_status("Timestamp file loaded.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load timestamps: {str(e)}")
                self.set_status("Timestamp load failed.")

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv")
        ])
        if self.video_path:
            messagebox.showinfo("Video Loaded", f"Selected video:\n{self.video_path}")
            self.set_status("Original video loaded.")
            
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

    def compute_freezing(self, timestamps):
        scorer = self.dlc_result.columns.levels[0][0]
        left_ear = 'left_ear'
        right_ear = 'right_ear'
        tail_base = 'tail_base'
        mid_back = 'mid_back'

        le_x = self.dlc_result.loc[:, (scorer, left_ear, 'x')].values.astype(float)
        le_y = self.dlc_result.loc[:, (scorer, left_ear, 'y')].values.astype(float)
        le_r = self.dlc_result.loc[:, (scorer, left_ear, 'likelihood')].values.astype(float)
        
        re_x = self.dlc_result.loc[:, (scorer, right_ear, 'x')].values.astype(float)
        re_y = self.dlc_result.loc[:, (scorer, right_ear, 'y')].values.astype(float)
        re_r = self.dlc_result.loc[:, (scorer, right_ear, 'likelihood')].values.astype(float)
        
        tb_x = self.dlc_result.loc[:, (scorer, tail_base, 'x')].values.astype(float)
        tb_y = self.dlc_result.loc[:, (scorer, tail_base, 'y')].values.astype(float)
        tb_r = self.dlc_result.loc[:, (scorer, tail_base, 'likelihood')].values.astype(float)

        mb_x = self.dlc_result.loc[:, (scorer, mid_back, 'x')].values.astype(float)
        mb_y = self.dlc_result.loc[:, (scorer, mid_back, 'y')].values.astype(float)
        mb_r = self.dlc_result.loc[:, (scorer, mid_back, 'likelihood')].values.astype(float)

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

        # dx = np.diff(centroid_x)
        # dy = np.diff(centroid_y)
        dx = np.diff(mb_x)
        dy = np.diff(mb_y)
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.insert(dist, 0, 0)
        raw_dist = dist

        dist[dist > 200] = np.nan
        valid = ~np.isnan(dist)
        dist_interp = np.interp(np.arange(len(dist)), np.flatnonzero(valid), dist[valid])

        kernel = np.ones(15) / 15
        speed_smooth = np.convolve(dist_interp, kernel, mode='same')

        # threshold = np.percentile(speed_smooth, 40)
        threshold = 1.20
        # print(threshold)

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

        start_time = timestamps['start_time'].iloc[0]
        end_time = timestamps['end_time'].iloc[-1]
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

        self.window_freeze = df

        self.raw_speed = raw_dist
        self.interp_speed = dist_interp
        self.smooth_speed = speed_smooth
        self.freeze_threshold = threshold
        self.freeze_idx = freeze_idx

        self.dlc_freeze_status = np.zeros(len(speed_smooth), dtype=int)
        self.dlc_freeze_status[freeze_idx] = 1

        return {'All': round(df['freeze_sec'].mean() * 100, 2) if not df.empty else 0}

    def compute_freezing_from_eztrack(self, fps):
        freezing_percentages = {}
        
        if hasattr(self, 'eztrack_result') and 'Freezing' in self.eztrack_result.columns:
            self.eztrack_freeze_status = (self.eztrack_result['Freezing'] == 100).astype(int).values
        else:
            self.eztrack_freeze_status = None
        
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
            end_frame = min((i + 1) * window_size_frames - 1, total_frames - 1)
            
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
        
        self.window_freeze_eztrack = pd.DataFrame({
            'time': window_times,
            'freeze_sec': freeze_sec
        })

    def plot_compare_raw_cleaning(self):
        if not hasattr(self, 'raw_speed'):
            messagebox.showwarning("No Data", "Please run DLC analysis first.")
            return

        x_time = np.arange(len(self.raw_speed)) / 30.0

        raw_speed = self.raw_speed.copy()
        valid = ~np.isnan(raw_speed)
        if valid.sum() >= 2:
            raw_speed = np.interp(np.arange(len(raw_speed)), np.flatnonzero(valid), raw_speed[valid])
        else:
            raw_speed[:] = 0

        plt.figure(figsize=(12, 4))

        for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
            state_rows = self.event_data[self.event_data['Event Type'] == state_code]
            for _, row in state_rows.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)

        plt.plot(x_time, self.raw_speed, alpha=0.6, label='Raw Speed')
        plt.plot(x_time, self.interp_speed, alpha=0.6, label='Interpolated')
        plt.plot(x_time, self.smooth_speed, linewidth=1.5, label='Smoothed')
        plt.axhline(self.freeze_threshold, linestyle='--', color="#ff0000ff", label='30th Percentile')

        in_freeze = np.zeros_like(self.smooth_speed)
        in_freeze[self.freeze_idx] = 1
        plt.fill_between(x_time, 0, max(self.smooth_speed), where=in_freeze > 0, color="#ff0000ff", alpha=0.3, label='Freezing')
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (px/frame)")
        plt.title("Speed Trace with Freezing Zones and State Background (DLC)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_freezing_windows_dlc(self):
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run DLC analysis first.")
            return

        plt.figure(figsize=(12, 5))

        for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
            state_rows = self.event_data[self.event_data['Event Type'] == state_code]
            for _, row in state_rows.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)

        df = self.window_freeze
        x = df['time'].values
        y = df['freeze_sec'].values
        plt.plot(x, y, marker='o', color='black', label='DLC Freezing')

        plt.xlabel("Time (s)")
        plt.ylabel("Freezing Duration (s) per 30s window")
        plt.title("Freezing Timeline with Background States (DLC)")
        plt.grid(False)
        plt.legend()
        max_x = np.ceil((x.max() + 15) / 30) * 30
        ticks = np.arange(0, max_x + 1, 30)
        plt.xticks(ticks)
        plt.tight_layout()
        plt.show()

    def plot_freezing_windows_eztrack(self):
        if not hasattr(self, 'window_freeze_eztrack') or self.window_freeze_eztrack is None:
            messagebox.showwarning("No Data", "Please run EZTrack analysis first.")
            return

        plt.figure(figsize=(12, 5))

        for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
            state_rows = self.event_data[self.event_data['Event Type'] == state_code]
            for _, row in state_rows.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)

        df = self.window_freeze_eztrack
        x = df['time'].values
        y = df['freeze_sec'].values
        plt.plot(x, y, marker='o', color='blue', label='EZTrack Freezing')

        plt.xlabel("Time (s)")
        plt.ylabel("Freezing Duration (s) per 30s window")
        plt.title("Freezing Timeline with Background States (EZTrack)")
        plt.grid(False)
        plt.legend()
        max_x = np.ceil((x.max() + 15) / 30) * 30
        ticks = np.arange(0, max_x + 1, 30)
        plt.xticks(ticks)
        plt.tight_layout()
        plt.show()

    def plot_compare_freezing(self):
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No DLC Data", "Please run DLC analysis first.")
            return
        if not hasattr(self, 'window_freeze_eztrack') or self.window_freeze_eztrack is None:
            messagebox.showwarning("No EZTrack Data", "Please run EZTrack analysis first.")
            return
        
        plt.figure(figsize=(12, 6))
        
        for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
            state_rows = self.event_data[self.event_data['Event Type'] == state_code]
            for _, row in state_rows.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)
        
        dlc_df = self.window_freeze
        dlc_x = dlc_df['time'].values
        dlc_y = dlc_df['freeze_sec'].values
        plt.plot(dlc_x, dlc_y, marker='o', color='black', label='DLC Freezing', linewidth=2)
        
        eztrack_df = self.window_freeze_eztrack
        eztrack_x = eztrack_df['time'].values
        eztrack_y = eztrack_df['freeze_sec'].values
        plt.plot(eztrack_x, eztrack_y, marker='s', color='blue', label='EZTrack Freezing', linewidth=2)
        
        all_x = np.concatenate([dlc_x, eztrack_x])
        min_x = np.floor(min(all_x) / 30) * 30
        max_x = np.ceil(max(all_x) / 30) * 30
        
        plt.xlabel("Time (s)")
        plt.ylabel("Freezing Duration (s) per 30s window")
        plt.title("DLC vs EZTrack Freezing Comparison")
        plt.grid(False)
        plt.legend()
        
        ticks = np.arange(min_x, max_x + 1, 30)
        plt.xticks(ticks)
        plt.xlim(min_x, max_x)
        
        plt.grid(False)
        
        plt.tight_layout()
        plt.show()

    def export_csv_dlc(self):
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run DLC analysis first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               title="Export Window Freezing Table (DLC)",
                                               filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        self.window_freeze.to_csv(save_path, index=False)
        messagebox.showinfo("Exported", f"Window-level results saved to:\n{save_path}")
        self.set_status("DLC window-wise results exported.")

    def export_csv_eztrack(self):
        if not hasattr(self, 'window_freeze_eztrack') or self.window_freeze_eztrack is None:
            messagebox.showwarning("No Data", "Please run EZTrack analysis first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               title="Export Window Freezing Table (EZTrack)",
                                               filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        self.window_freeze_eztrack.to_csv(save_path, index=False)
        messagebox.showinfo("Exported", f"EZTrack freezing results saved to:\n{save_path}")
        self.set_status("EZTrack results exported.")
    
    def show_video_dlc(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please load the original video first.")
            return
        if not hasattr(self, 'dlc_freeze_status'):
            messagebox.showwarning("No Analysis", "Please run DLC analysis first.")
            return
        
        self.method = "dlc"
        self.show_video()
    
    def show_video_eztrack(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please load the original video first.")
            return
        if not hasattr(self, 'eztrack_freeze_status'):
            messagebox.showwarning("No Analysis", "Please run EZTrack analysis first.")
            return
        
        self.method = "eztrack"
        self.show_video()
    
    def show_video(self):
        time_range = simpledialog.askstring("Time Range", 
                                          f"Enter start and end time in seconds (e.g., 30-60) [FPS: {self.fps:.2f}]:",
                                          parent=self.root)
        if not time_range:
            return
        
        try:
            start_time, end_time = map(float, time_range.split('-'))
            if start_time >= end_time:
                raise ValueError("Start time must be less than end time")
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid time format: {str(e)}")
            return
        
        self.start_frame = int(start_time * self.fps)
        self.end_frame = int(end_time * self.fps)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Video Error", "Failed to open video file.")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.end_frame > total_frames:
            self.end_frame = total_frames
            messagebox.showinfo("Adjusted Time", 
                              f"End time adjusted to {self.end_frame/self.fps:.2f}s (video end).")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        window_name = f"Freezing Analysis ({self.method.upper()})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        current_frame = self.start_frame
        while current_frame <= self.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if self.method == "dlc" and current_frame < len(self.dlc_freeze_status):
                status = self.dlc_freeze_status[current_frame]
            elif self.method == "eztrack" and current_frame < len(self.eztrack_freeze_status):
                status = self.eztrack_freeze_status[current_frame]
            else:
                status = 0
                
            status_text = "FREEZING" if status == 1 else "ACTIVE"
            text_color = (0, 0, 255) if status == 1 else (0, 255, 0)
            
            cv2.rectangle(frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Status: {status_text}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
            cv2.putText(frame, f"Time: {current_frame/self.fps:.2f}s | Frame: {current_frame}", 
                       (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Method: {self.method.upper()}", 
                       (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(int(1000/self.fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                while True:
                    key2 = cv2.waitKey(0)
                    if key2 == ord('p') or key2 == ord('q'):
                        break
                if key2 == ord('q'):
                    break
            
            current_frame += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        save = messagebox.askyesno("Save Video", 
                                  "Do you want to save this annotated video segment?")
        if save:
            self.save_annotated_video()

    def save_annotated_video(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")],
            title="Save Annotated Video"
        )
        
        if not save_path:
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Video Error", "Failed to open video file for saving.")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.end_frame > total_frames:
            self.end_frame = total_frames
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, self.fps, (frame_width, frame_height))
        
        if not out.isOpened():
            messagebox.showerror("Video Error", "Failed to create video writer.")
            cap.release()
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Saving Video")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)
        
        progress_label = ttk.Label(progress_window, text="Saving video...")
        progress_label.pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                      maximum=self.end_frame - self.start_frame)
        progress_bar.pack(fill='x', padx=20, pady=5)
        
        progress_window.update()
        
        current_frame = self.start_frame
        frame_count = 0
        total_frames_to_save = self.end_frame - self.start_frame
        
        while current_frame <= self.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if self.method == "dlc" and current_frame < len(self.dlc_freeze_status):
                status = self.dlc_freeze_status[current_frame]
            elif self.method == "eztrack" and current_frame < len(self.eztrack_freeze_status):
                status = self.eztrack_freeze_status[current_frame]
            else:
                status = 0
                
            status_text = "FREEZING" if status == 1 else "ACTIVE"
            text_color = (0, 0, 255) if status == 1 else (0, 255, 0)
            
            cv2.rectangle(frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Status: {status_text}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
            cv2.putText(frame, f"Time: {current_frame/self.fps:.2f}s | Frame: {current_frame}", 
                       (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Method: {self.method.upper()}", 
                       (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(frame)
            
            frame_count += 1
            progress_var.set(frame_count)
            progress_window.update()
            
            current_frame += 1
        
        cap.release()
        out.release()
        progress_window.destroy()
        
        messagebox.showinfo("Video Saved", f"Annotated video saved to:\n{save_path}")
        self.set_status(f"Annotated video saved for {self.method} method.")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = FreezingAnalyzerApp(root)
    root.mainloop()