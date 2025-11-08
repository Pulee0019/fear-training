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
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
from operator import itemgetter
from tkinter import ttk, filedialog, messagebox

class FreezingAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Conditioning Freezing Analyzer")
        self.root.geometry("200x350")
        self.root.resizable(False, False)

        self.dlc_results_path = ""
        self.timestamp_path = ""
        self.freezing_data = {}

        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky="NSEW")

        self.create_button(main_frame, "Load DLC Results Folder", self.load_DLC_Results, 0)
        self.create_button(main_frame, "Load Timestamp CSV", self.load_timestamps, 1)
        self.create_button(main_frame, "Run Freezing Analysis", self.analyze, 2)
        self.create_button(main_frame, "Export Results to CSV", self.export_csv, 3)
        self.create_button(main_frame, "Compare Raw vs Cleaned", self.plot_compare_raw_cleaning, 4)
        self.create_button(main_frame, "Plot Freezing Timeline", self.plot_freezing_windows, 5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="EW")

    def create_button(self, frame, text, command, row):
        btn = ttk.Button(frame, text=text, command=command)
        btn.grid(row=row, column=0, pady=6, sticky="EW")

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_DLC_Results(self):
        self.dlc_results_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.dlc_results_path:
            messagebox.showinfo("DLC results loaded", f"Selected model:\n{self.dlc_results_path}")
            self.set_status("DLC results loaded.")

    def load_timestamps(self):
        self.timestamp_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.timestamp_path:
            messagebox.showinfo("Timestamps Loaded", f"Selected timestamp file:\n{self.timestamp_path}")
            self.set_status("Timestamp file loaded.")

    def analyze(self):
        if not all([self.dlc_results_path, self.timestamp_path]):
            messagebox.showwarning("Missing Inputs", "Please load all required files first.")
            return

        self.set_status("Loading DLC result...")
        self.dlc_result = pd.read_csv(self.dlc_results_path, header=[0, 1, 2])

        self.set_status("Loading timestamps...")
        timestamps = pd.read_csv(self.timestamp_path)

        self.set_status("Computing freezing data...")
        self.freezing_data = self.compute_freezing(timestamps)
        messagebox.showinfo("Done", "Freezing analysis completed!")
        self.set_status("Analysis completed.")

    def compute_freezing(self, timestamps):
        scorer = self.dlc_result.columns.levels[0][0]
        mid_back = 'mid_back'

        x = self.dlc_result.loc[:, (scorer, mid_back, 'x')].values
        y = self.dlc_result.loc[:, (scorer, mid_back, 'y')].values

        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.insert(dist, 0, 0)
        raw_dist = dist

        dist[dist > 200] = np.nan
        valid = ~np.isnan(dist)
        dist_interp = np.interp(np.arange(len(dist)), np.flatnonzero(valid), dist[valid])

        kernel = np.ones(15) / 15
        speed_smooth = np.convolve(dist_interp, kernel, mode='same')

        # threshold = np.percentile(speed_smooth, 45)
        # print(threshold)

        threshold = 1.20

        freeze_flag = speed_smooth < threshold

        groups = groupby(enumerate(freeze_flag), key=lambda x: x[1])
        freeze_idx = []
        for k, g in groups:
            g = list(g)
            if k and len(g) >= 15:
                idxs = [i for i, _ in g]
                freeze_idx.extend(idxs)

        fps = 30
        window_sec = 30
        window_size = fps * window_sec

        start_time = timestamps['start_time'].iloc[0]
        end_time = timestamps['end_time'].iloc[-1]
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        segment = speed_smooth[start_frame:end_frame]

        result_vector = []
        time_vector = []

        for i in range(0, len(segment), window_size):
            window_idx = np.arange(start_frame + i, min(start_frame + i + window_size, end_frame))
            count = np.sum(np.isin(window_idx, freeze_idx))
            result_vector.append(count)
            time_vector.append((start_frame + i + min(len(window_idx) // 2, window_size // 2)) / fps)

        df = pd.DataFrame({
            'time': time_vector,
            'freeze_frame': result_vector,
            'freeze_sec': np.array(result_vector) / fps
        })

        self.window_freeze = df

        self.raw_speed = raw_dist
        self.interp_speed = dist_interp
        self.smooth_speed = speed_smooth
        self.freeze_threshold = threshold
        self.freeze_idx = freeze_idx
        self.mid_back_x = x
        self.mid_back_y = y

        return {'All': round(df['freeze_sec'].mean() * 100, 2) if not df.empty else 0}

    def plot_compare_raw_cleaning(self):
        if not hasattr(self, 'raw_speed'):
            messagebox.showwarning("No Data", "Please run the analysis first.")
            return

        x_time = np.arange(len(self.raw_speed)) / 30.0

        raw_speed = self.raw_speed.copy()
        valid = ~np.isnan(raw_speed)
        if valid.sum() >= 2:
            raw_speed = np.interp(np.arange(len(raw_speed)), np.flatnonzero(valid), raw_speed[valid])
        else:
            raw_speed[:] = 0

        plt.figure(figsize=(12, 4))

        timestamps = pd.read_csv(self.timestamp_path)
        for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
            state_rows = timestamps[timestamps['Event Type'] == state_code]
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
        plt.title("Speed Trace with Freezing Zones and State Background")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_freezing_windows(self):
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run the analysis first.")
            return

        timestamps = pd.read_csv(self.timestamp_path)

        plt.figure(figsize=(12, 5))

        for state_code, color in [(0, "#b6b6b6ff"), (1, "#fcb500f0")]:
            state_rows = timestamps[timestamps['Event Type'] == state_code]
            for _, row in state_rows.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=color, alpha=0.3)

        df = self.window_freeze
        x = df['time'].values
        y = df['freeze_sec'].values
        plt.plot(x, y, marker='o', color='black', label='Freezing')

        plt.xlabel("Time (s)")
        plt.ylabel("Freezing Duration (s) per 30s window")
        plt.title("Freezing Timeline with Background States")
        plt.grid(False)
        plt.legend()
        max_x = np.ceil((x.max() + 15) / 30) * 30
        ticks = np.arange(0, max_x + 1, 30)
        plt.xticks(ticks)
        plt.tight_layout()
        plt.show()

    def export_csv(self):
        if not hasattr(self, 'window_freeze') or self.window_freeze is None:
            messagebox.showwarning("No Data", "Please run the analysis first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Export Window Freezing Table")
        if not save_path:
            return

        self.window_freeze.to_csv(save_path, index=False)

        messagebox.showinfo("Exported", f"Window-level results saved to:\n{save_path}")
        self.set_status("Window-wise results exported.")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = FreezingAnalyzerApp(root)
    root.mainloop()