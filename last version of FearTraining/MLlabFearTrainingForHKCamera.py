# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 17:18:40 2025

@author: Pulee
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sounddevice as sd
import threading
import cv2
import time
import serial
import queue
from PIL import Image, ImageTk, ImageOps
from PyDAQmx import Task
import ctypes
import os
import csv
from ctypes import *
from CameraParams_header import *
from MvErrorDefine_const import *
from MvCameraControl_class import *

PixelType_Mono8 = 17301505  # You can change it according demand and PixelType_header.py

def list_audio_devices():
    devices = sd.query_devices()
    return [d['name'] for d in devices if d['max_output_channels'] > 0]

class FearTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Training Experiment")

        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.tlayer_type = MV_USB_DEVICE

        self.acq_queue = queue.Queue(maxsize=200)
        self.rec_queue = queue.Queue(maxsize=200)

        self.timeline = []
        self.previewing = False
        self.recording = False
        self.capture = None
        self.video_writer = None
        self.save_dir = os.getcwd()
        self.last_keypoints = None
        self.video_start_time = None
        self.event_log = []

        self.exposure_time = tk.DoubleVar(value=1000.0)
        self.resolution_choice = tk.StringVar(value="1280x1024")
        self.record_rate = tk.DoubleVar(value=30.0)
        self.sample_rate = tk.DoubleVar(value=90.0)

        self.pixel_type = PixelType_Mono8
        self.selected_audio_device = None
        self.user_filename_prefix = tk.StringVar(value="experiment")

        self.arduino_port = "COM10"
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            print("Arduino port opened.")
            time.sleep(2)
        except serial.SerialException as e:
            messagebox.showerror("Arduino Error", f"Failed to open {self.arduino_port}: {e}, if you don't need to control fiber photometry, please ignore it!")
            self.arduino = None

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def open(self):
        ret = self.cam.MV_CC_EnumDevices(self.tlayer_type, self.device_list)
        if ret != MV_OK or self.device_list.nDeviceNum == 0:
            raise RuntimeError("No camera found.")
    
        dev_info = self.device_list.pDeviceInfo[0].contents
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice()
    
        self.cam.MV_CC_SetEnumValue("PixelFormat", self.pixel_type)
    
        width_info = MVCC_INTVALUE()
        height_info = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("WidthMax", width_info)
        self.cam.MV_CC_GetIntValue("HeightMax", height_info)
    
        offsetx_info = MVCC_INTVALUE()
        offsety_info = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("OffsetX", offsetx_info)
        self.cam.MV_CC_GetIntValue("OffsetY", offsety_info)
    
        self.cam.MV_CC_SetIntValue("OffsetX", offsetx_info.nMin)
        self.cam.MV_CC_SetIntValue("OffsetY", offsety_info.nMin)
    
        self.cam.MV_CC_SetIntValue("Width", width_info.nCurValue)
        self.cam.MV_CC_SetIntValue("Height", height_info.nCurValue)
    
        # self.width = width_info.nCurValue
        # self.height = height_info.nCurValue
        
        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.sample_rate.get())
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time.get())
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImages)
        self.cam.MV_CC_StartGrabbing()
        print(f"[INFO] Opened camera with resolution {self.width} x {self.height}, offset 0,0")

    def close(self):
        try:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            print("[INFO] Camera successfully closed.")
        except Exception as e:
            print(f"[WARN] Error closing camera: {e}")

    def setup_ui(self):
        # Variables for block type and durations
        self.block_type = tk.StringVar(value="Wait")
        self.block_duration = tk.StringVar(value="5")
        # Create block type dropdown
        ttk.Label(self.root, text="Block Type:").grid(row=0, column=0)
        type_combo = ttk.Combobox(self.root, textvariable=self.block_type, values=["Wait", "Sound", "Sound+Shock"])
        type_combo.grid(row=0, column=1)
        # Create general duration field (for Wait and Sound types)
        self.label_duration = ttk.Label(self.root, text="Duration (s):")
        self.label_duration.grid(row=1, column=0)
        self.entry_duration = ttk.Entry(self.root, textvariable=self.block_duration)
        self.entry_duration.grid(row=1, column=1)
        # Create additional fields for Sound+Shock type (initially hidden)
        self.sound_duration = tk.StringVar(value="5")
        self.shock_lead = tk.StringVar(value="2")
        self.shock_duration = tk.StringVar(value="1")
        self.label_sound_duration = ttk.Label(self.root, text="Sound Duration (s):")
        self.entry_sound_duration = ttk.Entry(self.root, textvariable=self.sound_duration)
        self.label_shock_lead = ttk.Label(self.root, text="Shock Start (s before end):")
        self.entry_shock_lead = ttk.Entry(self.root, textvariable=self.shock_lead)
        # self.label_shock_duration = ttk.Label(self.root, text="Shock Duration (s):")
        # self.entry_shock_duration = ttk.Entry(self.root, textvariable=self.shock_duration)
        # Buttons to manage timeline blocks
        ttk.Button(self.root, text="Add Block", command=self.add_block).grid(row=4, column=0, columnspan=2)
        ttk.Button(self.root, text="Remove Selected Block", command=self.remove_selected_block).grid(row=5, column=0, columnspan=2)
        ttk.Button(self.root, text="Clear All Blocks", command=self.clear_blocks).grid(row=6, column=0, columnspan=2)
        # Filename prefix field
        ttk.Label(self.root, text="Filename Prefix:").grid(row=7, column=0)
        ttk.Entry(self.root, textvariable=self.user_filename_prefix).grid(row=7, column=1)
        # Timeline listbox to display sequence of blocks
        self.timeline_listbox = tk.Listbox(self.root, width=40)
        self.timeline_listbox.grid(row=0, column=2, rowspan=8, padx=10)
        self.timeline_listbox.bind("<<ListboxSelect>>", self.sync_selection)
        # Device selection and test buttons
        ttk.Button(self.root, text="Open Camera", command=self.open_camera).grid(row=8, column=0)
        ttk.Button(self.root, text="Open Speaker", command=self.select_devices).grid(row=8, column=1)
        ttk.Button(self.root, text="Test Speaker", command=self.test_speaker).grid(row=8, column=2)
        ttk.Button(self.root, text="Choose Save Directory", command=self.choose_directory).grid(row=9, column=0, columnspan=2)
        ttk.Button(self.root, text="Preview Camera", command=self.toggle_preview).grid(row=9, column=2)
        ttk.Button(self.root, text="Start Experiment", command=self.run_timeline).grid(row=10, column=2)
        # Video preview/recording label (hidden by default)
        self.video_label = tk.Label(root)
        self.video_label.grid(row=12, column=0, columnspan=3)
        self.video_label.grid_remove()
        # Bind combobox selection to show/hide fields based on block type
        type_combo.bind("<<ComboboxSelected>>", self.on_type_change)

        self.timeRecording = tk.Label(self.root, text="Recording: 00:00")
        self.timeRecording.grid(row=10, column=1)

    def on_type_change(self, event=None):
        # Adjust visible input fields based on selected block type
        btype = self.block_type.get()
        if btype == "Sound+Shock":
            # Hide single duration field
            self.label_duration.grid_remove()
            self.entry_duration.grid_remove()
            # Show Sound+Shock fields
            self.label_sound_duration.grid(row=1, column=0)
            self.entry_sound_duration.grid(row=1, column=1)
            self.label_shock_lead.grid(row=2, column=0)
            self.entry_shock_lead.grid(row=2, column=1)
            # self.label_shock_duration.grid(row=3, column=0)
            # self.entry_shock_duration.grid(row=3, column=1)
        else:
            # Hide Sound+Shock specific fields
            self.label_sound_duration.grid_remove()
            self.entry_sound_duration.grid_remove()
            self.label_shock_lead.grid_remove()
            self.entry_shock_lead.grid_remove()
            # self.label_shock_duration.grid_remove()
            # self.entry_shock_duration.grid_remove()
            # Show single duration field
            self.label_duration.grid(row=1, column=0)
            self.entry_duration.grid(row=1, column=1)

    def test_speaker(self):
        self.play_sound(1.0)

    def add_block(self):
        btype = self.block_type.get()
        if btype == "Sound+Shock":
            try:
                sound_dur = float(self.sound_duration.get())
                shock_lead = float(self.shock_lead.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for Sound+Shock parameters")
                return
            if sound_dur <= 0 or shock_lead < 0:
                messagebox.showerror("Error", "Sound duration and shock duration must be positive, and shock start time must be non-negative")
                return
            if shock_lead > sound_dur:
                messagebox.showerror("Error", "Shock start time must be less than or equal to sound duration")
                return
            self.timeline.append((btype, sound_dur, shock_lead))
            self.timeline_listbox.insert(tk.END, f"{btype} - Sound {sound_dur:.1f}s, Shock start {shock_lead:.1f}s before end")
        else:
            try:
                dur = float(self.block_duration.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid duration")
                return
            self.timeline.append((btype, dur))
            self.timeline_listbox.insert(tk.END, f"{btype} - {dur:.1f}s")

    def remove_selected_block(self):
        sel = self.timeline_listbox.curselection()
        if sel:
            index = sel[0]
            del self.timeline[index]
            self.timeline_listbox.delete(index)

    def clear_blocks(self):
        self.timeline.clear()
        self.timeline_listbox.delete(0, tk.END)

    def sync_selection(self, event):
        sel = self.timeline_listbox.curselection()
        if sel:
            index = sel[0]
            entry = self.timeline[index]
            btype = entry[0]
            self.block_type.set(btype)
            # Update field visibility for selected type
            self.on_type_change()
            if btype == "Sound+Shock":
                # entry format: (type, sound_dur, shock_lead)
                _, sound_dur, shock_lead = entry
                self.sound_duration.set(str(sound_dur))
                self.shock_lead.set(str(shock_lead))
                #self.shock_duration.set(str(shock_dur))
            else:
                dur = entry[1]
                self.block_duration.set(str(dur))

    def choose_directory(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.save_dir = selected_dir

    def select_devices(self):
        mic_list = list_audio_devices()

        def apply_selection():
            self.selected_audio_device = mic_combo.get()
            sel_win.destroy()

        sel_win = tk.Toplevel(self.root)
        sel_win.title("Select Devices")

        tk.Label(sel_win, text="Select Speaker:").pack()
        mic_combo = ttk.Combobox(sel_win, values=mic_list)
        mic_combo.pack()
        mic_combo.current(0)

        ttk.Button(sel_win, text="OK", command=apply_selection).pack(pady=10)

    def open_camera(self):
        popup = tk.Toplevel()
        popup.title("Camera Settings")

        tk.Label(popup, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        exposure_var = tk.DoubleVar(value=self.exposure_time.get())
        tk.Entry(popup, textvariable=exposure_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(popup, text="Resolution:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        resolution_var = tk.StringVar(value=self.resolution_choice.get())
        resolution_menu = ttk.OptionMenu(popup, resolution_var, resolution_var.get(), "1280x1024", "1280x720", "640x480")
        resolution_menu.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(popup, text="Acquisition FPS:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        sample_rate_var = tk.DoubleVar(value=self.sample_rate.get())
        tk.Entry(popup, textvariable=sample_rate_var).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(popup, text="Recording FPS:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        record_rate_var = tk.DoubleVar(value=self.record_rate.get())
        tk.Entry(popup, textvariable=record_rate_var).grid(row=3, column=1, padx=5, pady=5)

        def apply_params():
            try:
                self.exposure_time.set(exposure_var.get())
                self.resolution_choice.set(resolution_var.get())
                self.sample_rate.set(sample_rate_var.get())
                self.record_rate.set(record_rate_var.get())

                res_text = resolution_var.get()
                width, height = map(int, res_text.split("x"))
                self.width = width
                self.height = height
                self.frame_rate = sample_rate_var.get()
                self.record_rate = record_rate_var.get()
                
                self.close()
                self.open()
                self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_var.get())
                self.cam.MV_CC_SetIntValue("Width", width)
                self.cam.MV_CC_SetIntValue("Height", height)
                self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", sample_rate_var.get())
                messagebox.showinfo("Success", "Camera parameters applied and opened.")

                popup.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply parameters: {e}")

        ttk.Button(popup, text="Apply Settings and Open Camera", command=apply_params).grid(row=4, column=0, columnspan=2, pady=10)

    def toggle_preview(self):
        if not self.previewing:
            try:
                self.open()
                self.previewing = True
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to start preview: {str(e)}")
        else:
            self.previewing = False

    def update_preview(self):
        if self.previewing:
            try:
                stFrame = MV_FRAME_OUT()
                ret = self.cam.MV_CC_GetImageBuffer(stFrame, 100)
                if ret == MV_OK:
                    buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                    img_np = np.frombuffer(buf, dtype=np.uint8)
                    img_np = img_np.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))

                    pil_image = Image.fromarray(img_np)

                    target_size = (640, 480)
                    padded = ImageOps.pad(pil_image.convert("L"), target_size, color=0, centering=(0.5, 0.5))
                    imgtk = ImageTk.PhotoImage(padded)
                    
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                    
                    self.cam.MV_CC_FreeImageBuffer(stFrame)
            except Exception as e:
                print(f"Preview error: {str(e)}")

            self.root.after(50, self.update_preview)

    def update_timer(self):

        current = self.timeRecording['text']
        minutes, seconds = map(int, current[-5:].split(':'))

        seconds += 1
        if seconds == 60:
            seconds = 0
            minutes += 1
        formatted_time = f"{minutes:02d}:{seconds:02d}"

        self.timeRecording.config(text=f"Recording: {formatted_time}")

        self.timer_id = self.timeRecording.after(1000, self.update_timer)

    def stop_timer(self):
        self.timeRecording.after_cancel(self.timer_id)
        self.timeRecording.config(text=f"Recording: 00:00")

    def run_timeline(self):
        if not self.timeline:
            messagebox.showerror("Error", "Timeline is empty")
            return
        threading.Thread(target=self._run_timeline).start()

    def _run_timeline(self):
        self.event_log = []
        self.start_video_recording(self.save_dir)
        self.video_start_time = time.time()
        self.update_timer()

        '''
        self.total_time = 0
        for block in self.timeline:
            self.total_time += block[1]
        self.progressLabel = ttk.Label(self.root, text="Progress:     ")
        self.progressLabel.grid(row=10, column=0)
        self.progressbar = ttk.Progressbar(self.root, length=400, mode='determinate', maximum=self.total_time)
        self.progressbar.grid(row=10, column=1)
        '''

        i = 0
        while i < len(self.timeline):
            entry = self.timeline[i]
            btype = entry[0]
            start_time = time.time() - self.video_start_time

            if btype == "Wait":
                dur = entry[1]
                #self.progressLabel.config(text=f'Progress: {btype}')
                #self.root.update_idletasks()
                time.sleep(dur)
                #self.progressbar["value"] += dur
                #self.root.update_idletasks()
                end_time = time.time() - self.video_start_time
                self.event_log.append((start_time, end_time, 0))
            elif btype == "Sound":
                dur = entry[1]
                #self.progressLabel.config(text=f'Progress: {btype}')
                #self.root.update_idletasks()
                self.play_sound(dur)
                #self.progressbar["value"] += dur
                #self.root.update_idletasks()
                end_time = time.time() - self.video_start_time
                self.event_log.append((start_time, end_time, 1))
            elif btype == "Sound+Shock":
                sound_dur = entry[1]
                shock_lead = entry[2]
                #self.progressLabel.config(text=f'Progress: {btype}')
                #self.root.update_idletasks()
                # shock_dur = entry[3]
                # Calculate shock timing relative to sound end
                shock_start = max(0, sound_dur - shock_lead)
                shock_end = shock_start + shock_lead
                # Play sound (non-blocking) for the entire sound duration
                sample_rate = int(4 * 4000)
                t = np.arange(0, sound_dur, 1.0 / sample_rate)
                sound_wave = 0.5 * np.sin(2 * np.pi * 4000 * t)
                sd.play(sound_wave, samplerate=sample_rate, device=self.selected_audio_device, latency='low')
                # Wait until the shock should start
                if shock_start > 0:
                    time.sleep(shock_start)
                # Trigger electrical stimulation (shock)
                self.trigger_stimulation(shock_lead)
                #self.progressbar["value"] += sound_dur + shock_lead
                #self.root.update_idletasks()
                # Wait for the sound to finish playing (if not already finished)
                sd.wait()
                # Record end time after both sound and shock are done
                end_time = time.time() - self.video_start_time
                # Log sound and shock events separately
                self.event_log.append((start_time, end_time, 1))   # Sound event
                self.event_log.append((start_time + shock_start, end_time, 2))  # Shock event
            else:
                end_time = time.time() - self.video_start_time
                self.event_log.append((start_time, end_time, -1))
            i += 1
        
        self.stop_timer()
        self.stop_video_recording()
        self.save_event_log()
        print("Experiment complete")

    def play_sound(self, duration, freq=4000, amp=0.5):
        sample_rate = 4 * freq
        t = np.arange(0, duration, 1 / sample_rate)
        sound_wave = amp * np.sin(2 * np.pi * freq * t)
        sd.play(sound_wave, samplerate=sample_rate, device=self.selected_audio_device)
        sd.wait()

    def trigger_stimulation(self, duration):
        try:
            task = Task()
            task.CreateAOVoltageChan("Dev1/ao0", "", -10.0, 10.0, 10348, None)
            sampling_rate = 5000
            total_samples = int(sampling_rate * duration)
            if total_samples <= 0:
                total_samples = int(sampling_rate * 0.1)
            output_data = np.full(total_samples, 5.0, dtype=np.float64)
            task.CfgSampClkTiming("", sampling_rate, 10280, 10178, total_samples)
            written = ctypes.c_int32()
            task.WriteAnalogF64(total_samples, False, 10.0, 0, output_data, ctypes.byref(written), None)
            task.StartTask()
            task.WaitUntilTaskDone(max(10.0, duration))
            task.StopTask()
            task.ClearTask()
            reset_task = Task()
            reset_task.CreateAOVoltageChan("Dev1/ao0", "", -10.0, 10.0, 10348, None)
            reset_task.StartTask()
            reset_task.WriteAnalogScalarF64(True, 0.001, 0.0, None)
            reset_task.StopTask()
            reset_task.ClearTask()
            print("Shock delivered")
        except Exception as e:
            print(f"Stimulation error: {e}")

    def start_video_recording(self, save_path):
        self.running = True
        self.recording = True
        self.save_path = self.save_dir
        
        param = MV_CC_RECORD_PARAM()
        param.enPixelType = self.pixel_type
        param.nWidth = self.width
        param.nHeight = self.height
        param.fFrameRate = self.record_rate
        param.nBitRate = 8192
        param.enRecordFmtType = 1
        param.strFilePath = c_char_p(save_path.encode('utf-8'))
        self.cam.MV_CC_StartRecord(param)
        
        threading.Thread(target=self._acquisition_loop, daemon=True).start()
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _acquisition_loop(self):
        stFrame = MV_FRAME_OUT()
        while self.running:
            t0 = time.time()
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == MV_OK:
                buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                self.last_frame_data = buf
                h = stFrame.stFrameInfo.nHeight
                w = stFrame.stFrameInfo.nWidth
                expected_size = h * w

                img_np = np.frombuffer(buf, dtype=np.uint8)

                if img_np.size >= expected_size:
                    img_np = img_np[:expected_size].reshape((h, w))
                else:
                    print(f"[ERROR] Received frame size {img_np.size}, expected {expected_size}. Skipping frame.")
                    continue

                self.cam.MV_CC_FreeImageBuffer(stFrame)

                while not self.acq_queue.empty():
                    try:
                        self.acq_queue.get_nowait()
                    except queue.Empty:
                        break

                self.acq_queue.put((img_np.copy(), time.time()))

                self.acqDuration.append(time.time() - t0)
            else:
                time.sleep(0.005)
    
    def _record_loop(self):
        stInput = MV_CC_INPUT_FRAME_INFO()
        self.frame_counter = 0
        start_time = time.time()
        self.mark_ttl_on()
        target_interval_t = 1.0 / self.record_rate  # Target interval for recording frame rate

        while self.recording:
            record_loop_start = time.time()
    
            if self.last_frame_data is None:
                time.sleep(0.01)
                continue
    
            stInput.pData = cast(c_char_p(self.last_frame_data), POINTER(c_ubyte))
            stInput.nDataLen = len(self.last_frame_data)
            self.cam.MV_CC_InputOneFrame(stInput)
            self.frame_counter += 1
    
            if self.frame_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = self.frame_counter / elapsed
                self.gui_fps_var.set(f"FPS: {fps:.2f}")
    
            target_interval = self.frame_counter * target_interval_t - (time.time() - start_time)
            if target_interval > 0:
                time.sleep(target_interval)
            record_loop_end = time.time()
            self.recDuration.append(record_loop_end - record_loop_start)

    def stop_video_recording(self):
        self.running = False
        self.recording = False
        self.mark_ttl_off()
        self.cam.MV_CC_StopRecord()

    def save_event_log(self):
        if not self.video_start_time or not self.event_log:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.user_filename_prefix.get()}_events_{timestamp}.csv"
        filepath = os.path.join(self.save_dir, filename)
        try:
            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["start_time", "end_time", "Event Type"])
                for start, end, event in self.event_log:
                    writer.writerow([f"{start:.3f}", f"{end:.3f}", event])
            print(f"Event log saved: {filepath}")
        except Exception as e:
            print(f"Failed to save event log: {e}")

    def mark_ttl_on(self):
        self._send_to_arduino('T')
        print('mark on')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

    def on_closing(self):
        self.previewing = False
        self.stop_video_recording()
        self.close()
        
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")
        
        self.root.destroy()


    # def on_closing(self):
    #     try:
    #         if self.arduino and self.arduino.is_open:
    #             self.arduino.close()
    #             print("Arduino port closed.")
    #     except Exception as e:
    #         print(f"[ERROR] Exception during closing: {e}")
    #     finally:
    #         self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FearTrainingGUI(root)
    root.mainloop()
