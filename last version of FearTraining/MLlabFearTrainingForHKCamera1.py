# -*- coding: utf-8 -*-
"""
Created on Tue Jul 8 19:18:40 2025

@author: Pulee
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sounddevice as sd
import threading
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

class HKCamera:
    def __init__(self, root, arduino=None):
        self.root = root
        self.arduino = arduino

        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.tlayer_type = MV_USB_DEVICE

        self.recording = False
        self.previewing = False
        self.running = False
        self.preview_callback = None
        self.save_path = None

        self.pixel_type = PixelType_Mono8
        self.last_frame_data = None
        self.frame_counter = 0
        self.acq_queue = queue.Queue(maxsize=200)
        self.acqDuration = []
        self.recDuration = []

        self.width = 1280
        self.height = 1024
        self.record_rate = 30.0
        self.sample_rate = 90.0
        self.exposure_time = 1000.0

    def open(self):
        ret = self.cam.MV_CC_EnumDevices(self.tlayer_type, self.device_list)
        if ret != MV_OK or self.device_list.nDeviceNum == 0:
            raise RuntimeError("No camera found.")
    
        dev_info = self.device_list.pDeviceInfo[0].contents
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice()
    
        self.cam.MV_CC_SetEnumValue("PixelFormat", self.pixel_type)
    
        # 设置分辨率
        self.cam.MV_CC_SetIntValue("Width", self.width)
        self.cam.MV_CC_SetIntValue("Height", self.height)
        
        # 设置相机参数
        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.sample_rate)
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImages)
        self.cam.MV_CC_StartGrabbing()
        print(f"[INFO] Opened camera with resolution {self.width}x{self.height} @ {self.sample_rate}fps")

    def close(self):
        try:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            print("[INFO] Camera successfully closed.")
        except Exception as e:
            print(f"[WARN] Error closing camera: {e}")
    
    def set_parameters(self, width, height, exposure_time, sample_rate, record_rate):
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.sample_rate = sample_rate
        self.record_rate = record_rate

    def start_preview(self, callback):
        if self.previewing:
            print("[WARN] Preview already running.")
            return
        self.previewing = True
        self.preview_callback = callback
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def stop_preview(self):
        self.previewing = False

    def _preview_loop(self):
        stFrame = MV_FRAME_OUT()
        while self.previewing:
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == MV_OK:
                buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                img_np = np.frombuffer(buf, dtype=np.uint8).reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))

                pil_img = Image.fromarray(img_np)
                self.cam.MV_CC_FreeImageBuffer(stFrame)
                
                if self.preview_callback:
                    self.root.after(0, lambda: self.preview_callback(pil_img))
            else:
                time.sleep(0.01)

    def start_record(self, save_path):
        self.running = True
        self.recording = True
        self.save_path = save_path
        self.frame_counter = 0

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

    def stop_record(self):
        self.running = False
        self.recording = False
        self.mark_ttl_off()
        self.cam.MV_CC_StopRecord()
    
    def mark_ttl_on(self):
        self._send_to_arduino('T')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
    def trigger_laser(self):
        self._send_to_arduino('L')
    
    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

class FearTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fear Training Experiment")
        self.root.geometry("1200x800")

        self.arduino_port = "COM10"
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            print("Arduino port opened.")
            time.sleep(2)
        except serial.SerialException as e:
            messagebox.showerror("Arduino Error", f"Failed to open {self.arduino_port}: {e}, if you don't need to control fiber photometry, please ignore it!")
            self.arduino = None
        
        self.camera = HKCamera(root, arduino=self.arduino)
        self.camera.gui_fps_var = tk.StringVar(value="FPS: --")
        
        self.timeline = []
        self.save_dir = os.getcwd()
        self.video_start_time = None
        self.event_log = []

        self.exposure_time = tk.DoubleVar(value=1000.0)
        self.resolution_choice = tk.StringVar(value="1280x1024")
        self.record_rate = tk.DoubleVar(value=30.0)
        self.sample_rate = tk.DoubleVar(value=90.0)

        self.pixel_type = PixelType_Mono8
        self.selected_audio_device = None
        self.user_filename_prefix = tk.StringVar(value="experiment")

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(main_frame, text="Block Configuration")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame, text="Block Type:").grid(row=0, column=0, sticky="w", pady=2)
        self.block_type = tk.StringVar(value="Wait")
        type_combo = ttk.Combobox(control_frame, textvariable=self.block_type, 
                                 values=["Wait", "Sound", "Sound+Shock"], width=15)
        type_combo.grid(row=0, column=1, sticky="ew", pady=2)

        self.duration_frame = ttk.Frame(control_frame)
        self.duration_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        
        ttk.Label(self.duration_frame, text="Duration (s):").pack(side=tk.LEFT)
        self.block_duration = tk.StringVar(value="5")
        ttk.Entry(self.duration_frame, textvariable=self.block_duration, width=8).pack(side=tk.LEFT, padx=5)

        self.shock_frame = ttk.Frame(control_frame)
        
        ttk.Label(self.shock_frame, text="Sound Duration (s):").grid(row=0, column=0, sticky="w")
        self.sound_duration = tk.StringVar(value="5")
        ttk.Entry(self.shock_frame, textvariable=self.sound_duration, width=8).grid(row=0, column=1)
        
        ttk.Label(self.shock_frame, text="Shock Duration (s):").grid(row=1, column=0, sticky="w")
        self.shock_lead = tk.StringVar(value="2")
        ttk.Entry(self.shock_frame, textvariable=self.shock_lead, width=8).grid(row=1, column=1)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Add Block", command=self.add_block, width=12).pack(side=tk.TOP, pady=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected_block, width=12).pack(side=tk.TOP, pady=2)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_blocks, width=12).pack(side=tk.TOP, pady=2)

        ttk.Label(control_frame, text="Filename Prefix:").grid(row=3, column=0, sticky="w", pady=(10, 2))
        ttk.Entry(control_frame, textvariable=self.user_filename_prefix).grid(row=3, column=1, sticky="ew", pady=(10, 2))
        ttk.Button(control_frame, text="Choose Save Directory", command=self.choose_directory).grid(row=4, column=0, columnspan=2, pady=5)

        timeline_frame = ttk.LabelFrame(main_frame, text="Timeline Sequence")
        timeline_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.timeline_listbox = tk.Listbox(timeline_frame, font=("Arial", 10))
        scrollbar = ttk.Scrollbar(timeline_frame, orient="vertical", command=self.timeline_listbox.yview)
        self.timeline_listbox.configure(yscrollcommand=scrollbar.set)
        self.timeline_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.timeline_listbox.bind("<<ListboxSelect>>", self.sync_selection)

        device_frame = ttk.LabelFrame(main_frame, text="Device Control")
        device_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(device_frame, text="Select Camera", command=self.open_camera).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="Select Speaker", command=self.select_devices).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(device_frame, text="Test Speaker", command=self.test_speaker).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(device_frame, text="Start Preview", command=self.toggle_preview).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="Stop Preview", command=self.stop_preview).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="START EXPERIMENT", command=self.run_timeline, 
                  style="Accent.TButton").grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="EXIT", command=self.exit, 
                  style="Accent.TButton").grid(row=3, column=1, padx=5, pady=5)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.recording_label = ttk.Label(status_frame, text="Status: Ready")
        self.recording_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(status_frame, text="00:00:00")
        self.time_label.pack(side=tk.LEFT, padx=20)
        
        self.fps_label = ttk.Label(status_frame, textvariable=self.camera.gui_fps_var)
        self.fps_label.pack(side=tk.LEFT)

        self.video_frame = ttk.LabelFrame(self.root, text="Camera Preview")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.root.style = ttk.Style()
        self.root.style.configure("Accent.TButton", font=("Arial", 10, "bold"), foreground="black", background="#4CAF50")

        type_combo.bind("<<ComboboxSelected>>", self.on_type_change)

    def on_type_change(self, event=None):
        # Adjust visible input fields based on selected block type
        btype = self.block_type.get()
        if btype == "Sound+Shock":
            # Hide single duration field
            self.duration_frame.grid_remove()
            # Show Sound+Shock fields
            self.shock_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        else:
            # Hide Sound+Shock specific fields
            self.shock_frame.grid_remove()
            # Show single duration field
            self.duration_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

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
        sel_win.title("Select Speaker")

        tk.Label(sel_win, text="Select Speaker:").pack()
        mic_combo = ttk.Combobox(sel_win, values=mic_list, width=40)
        mic_combo.pack(padx=10, pady=5)
        if mic_list:
            mic_combo.current(0)

        ttk.Button(sel_win, text="OK", command=apply_selection).pack(pady=10)

    def open_camera(self):
        popup = tk.Toplevel(self.root)
        popup.title("Camera Settings")
        popup.resizable(False, False)

        ttk.Label(popup, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        exposure_var = tk.DoubleVar(value=self.exposure_time.get())
        ttk.Entry(popup, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Resolution:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        resolution_var = tk.StringVar(value=self.resolution_choice.get())
        resolution_menu = ttk.OptionMenu(popup, resolution_var, resolution_var.get(), 
                                       "1280x1024", "1280x720", "640x480")
        resolution_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(popup, text="Acquisition FPS:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        sample_rate_var = tk.DoubleVar(value=self.sample_rate.get())
        ttk.Entry(popup, textvariable=sample_rate_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Recording FPS:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        record_rate_var = tk.DoubleVar(value=self.record_rate.get())
        ttk.Entry(popup, textvariable=record_rate_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        def apply_params():
            try:
                self.exposure_time.set(exposure_var.get())
                self.resolution_choice.set(resolution_var.get())
                self.sample_rate.set(sample_rate_var.get())
                self.record_rate.set(record_rate_var.get())

                res_text = resolution_var.get()
                width, height = map(int, res_text.split("x"))

                self.camera.set_parameters(
                    width=width,
                    height=height,
                    exposure_time=exposure_var.get(),
                    sample_rate=sample_rate_var.get(),
                    record_rate=record_rate_var.get()
                )

                self.camera.close()
                self.camera.open()
                
                messagebox.showinfo("Success", "Camera parameters applied and opened.")
                popup.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply parameters: {e}")

        btn_frame = ttk.Frame(popup)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Apply Settings", command=apply_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=popup.destroy).pack(side=tk.LEFT, padx=5)

    def toggle_preview(self):
        if not self.camera.previewing:
            try:
                self.camera.start_preview(self.update_image)
                self.recording_label.config(text="Status: Previewing")
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to start preview: {str(e)}")
        else:
            self.camera.stop_preview()
            self.recording_label.config(text="Status: Ready")

    def stop_preview(self):
        self.camera.stop_preview()
        self.recording_label.config(text="Status: Preview Stopped")

    def update_image(self, pil_image):
        target_size = (640, 480)
        padded = ImageOps.pad(pil_image.convert("L"), target_size, color=0, centering=(0.5, 0.5))
        imgtk = ImageTk.PhotoImage(padded)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_timer(self):
        if self.camera.recording:
            elapsed = time.time() - self.video_start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            self.time_label.config(text=f"{hours:02d}:{mins:02d}:{secs:02d}")
            self.root.after(1000, self.update_timer)

    def run_timeline(self):
        if not self.timeline:
            messagebox.showerror("Error", "Timeline is empty")
            return
        threading.Thread(target=self._run_timeline, daemon=True).start()

    def _run_timeline(self):
        self.recording_label.config(text="Status: Running Experiment")
        self.event_log = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.user_filename_prefix.get()}_{timestamp}.avi"
        save_path = os.path.join(self.save_dir, filename)

        self.camera.start_record(save_path)
        self.video_start_time = time.time()
        self.update_timer()

        i = 0
        while i < len(self.timeline) and self.camera.recording:
            entry = self.timeline[i]
            btype = entry[0]
            start_time = time.time() - self.video_start_time

            if btype == "Wait":
                dur = entry[1]
                time.sleep(dur)
                end_time = time.time() - self.video_start_time
                self.event_log.append((start_time, end_time, 0))
            elif btype == "Sound":
                dur = entry[1]
                self.play_sound(dur)
                end_time = time.time() - self.video_start_time
                self.event_log.append((start_time, end_time, 1))
            elif btype == "Sound+Shock":
                sound_dur = entry[1]
                shock_lead = entry[2]
                shock_start = max(0, sound_dur - shock_lead)

                sample_rate = int(4 * 4000)
                t = np.arange(0, sound_dur, 1.0 / sample_rate)
                sound_wave = 0.5 * np.sin(2 * np.pi * 4000 * t)
                sd.play(sound_wave, samplerate=sample_rate, device=self.selected_audio_device, latency='low')

                if shock_start > 0:
                    time.sleep(shock_start)

                self.trigger_stimulation(shock_lead)

                sd.wait()

                end_time = time.time() - self.video_start_time
                self.event_log.append((start_time, end_time, 1))
                self.event_log.append((start_time + shock_start, end_time, 2))
            
            i += 1

        self.camera.stop_record()
        self.save_event_log()
        self.recording_label.config(text="Status: Experiment Complete")
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

    def on_closing(self):
        self.camera.stop_preview()
        # self.camera.stop_record()
        self.camera.close()

        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")

        self.root.destroy()
    
    def exit(self):
        self.camera.stop_preview()
        self.camera.close()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FearTrainingGUI(root)
    root.mainloop()