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
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import cv2
import os
import csv
import ctypes
import win32com.client
from ctypes import *
from PyDAQmx import Task
from PyDAQmx import *
from PyDAQmx.DAQmxTypes import DAQmxEveryNSamplesEventCallbackPtr
from CameraParams_header import *
from MvErrorDefine_const import *
from MvCameraControl_class import *
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PyDAQmx.DAQmxTypes import DAQmxEveryNSamplesEventCallbackPtr
from functools import partial

PixelType_Mono8 = 17301505

class SpeedSensor:
    """集成速度传感器功能"""
    def __init__(self, root):
        self.root = root
        self.running = False
        self.task = None
        self.file = None
        self.input_rate = 1000
        self.scale = 2 ** 15 / 10
        self.channels = ['Dev1/ai0']  # 默认使用第一个通道
        self.start_time = None
        
    def start_tracking(self, file_path):
        if self.running:
            return False
            
        try:
            self.file = open(file_path, 'wb')
        except IOError as e:
            messagebox.showerror("Error", f"Could not open file: {e}")
            return False
            
        try:
            self.task = Task()
            for ch in self.channels:
                self.task.CreateAIVoltageChan(
                    ch, "", DAQmx_Val_Cfg_Default, 
                    -10.0, 10.0, DAQmx_Val_Volts, None
                )
                
            self.task.CfgSampClkTiming(
                "", self.input_rate, DAQmx_Val_Rising, 
                DAQmx_Val_ContSamps, 100000
            )
            self.callback = DAQmxEveryNSamplesEventCallbackPtr(self.listener_callback)
            self.task.RegisterEveryNSamplesEvent(
                DAQmx_Val_Acquired_Into_Buffer, 100, 0, 
                self.callback, None
            )
            self.task.StartTask()
            self.running = True
            self.start_time = time.time()
            return True
        except Exception as e:
            messagebox.showerror("DAQ Error", f"Failed to start acquisition: {e}")
            if self.file:
                self.file.close()
                self.file = None
            return False

    def listener_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        try:
            read = ctypes.c_int32()
            buffer = np.zeros((len(self.channels), number_of_samples))
            
            self.task.ReadAnalogF64(
                number_of_samples, 10.0, DAQmx_Val_GroupByChannel,
                buffer, buffer.size, ctypes.byref(read), None
            )
            samples = read.value
            
            if samples > 0 and self.file:
                scaled = (buffer * self.scale).astype(np.int16)
                self.file.write(scaled.tobytes())
            return 0
        except Exception as e:
            print(f"Error in callback: {e}")
            return -1

    def stop_tracking(self):
        if not self.running: 
            return
            
        if self.task:
            try:
                self.task.StopTask()
                self.task.ClearTask()
            except Exception as e:
                print(f"Error stopping task: {e}")
            finally:
                self.task = None
                
        if self.file and not self.file.closed:
            self.file.close()
            self.file = None
            
        self.running = False
        return self.start_time, time.time() if self.start_time else (0, 0)

def get_camera_names():
    """
    Get the friendly names of all connected video capture devices.
    """
    wmi = win32com.client.GetObject("winmgmts:")
    cameras = wmi.InstancesOf("Win32_PnPEntity")
    camera_list = []
    for camera in cameras:
        if camera.Service == "usbvideo":  # Only list USB video devices
            camera_list.append(camera.Name)
    return camera_list

def is_camera_available(index):
    """
    Robustly check if a camera is available and return (True/False, device name).
    """
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return False, None
        ret, _ = cap.read()
        cap.release()
        if ret:
            names = get_camera_names()
            if index < len(names):
                return True, names[index]
            else:
                return True, f"Camera {index}"
        else:
            return False, None
    except Exception as e:
        print(f"[Camera {index}] check failed: {e}")
        return False, None

def list_available_cameras(max_devices=10):
    """
    List available and accessible camera device names up to a maximum number.
    """
    available = []
    for i in range(max_devices):
        success, name = is_camera_available(i)
        if success and name:
            available.append(name)
    return available

def list_audio_devices():
    devices = sd.query_devices()
    return [d['name'] for d in devices if d['max_output_channels'] > 0]

class HKCamera:
    def __init__(self, root, arduino=None, camera_id=0):
        self.root = root
        self.arduino = arduino
        self.camera_id = camera_id
        self.display_name = f"Camera {camera_id+1} (HIK)"

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
        
        self.start_time = None
        self.end_time = None
        self.current_fps = 0.0
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def open(self):
        ret = self.cam.MV_CC_EnumDevices(self.tlayer_type, self.device_list)
        if ret != MV_OK or self.device_list.nDeviceNum == 0:
            raise RuntimeError("No camera found.")
    
        if self.camera_id >= self.device_list.nDeviceNum:
            raise RuntimeError(f"Camera index {self.camera_id} not available")
            
        dev_info = self.device_list.pDeviceInfo[self.camera_id].contents
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice()
    
        self.cam.MV_CC_SetEnumValue("PixelFormat", self.pixel_type)

        self.cam.MV_CC_SetIntValue("Width", self.width)
        self.cam.MV_CC_SetIntValue("Height", self.height)

        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.sample_rate)
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImages)
        self.cam.MV_CC_StartGrabbing()
        print(f"[INFO] Opened camera {self.camera_id} with resolution {self.width}x{self.height} @ {self.record_rate}fps")

    def close(self):
        try:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            print(f"[INFO] Camera {self.camera_id} successfully closed.")
        except Exception as e:
            print(f"[WARN] Error closing camera {self.camera_id}: {e}")
    
    def set_parameters(self, width, height, exposure_time, sample_rate, record_rate):
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.sample_rate = sample_rate
        self.record_rate = record_rate

    def start_preview(self, callback):
        if self.previewing:
            print(f"[WARN] Camera {self.camera_id} preview already running.")
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

                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
                self.cam.MV_CC_FreeImageBuffer(stFrame)
                
                if self.preview_callback:
                    self.root.after(0, lambda img=pil_img: self.preview_callback(img, self.camera_id))
            else:
                time.sleep(0.01)

    def start_record(self, save_path):
        self.running = True
        self.recording = True
        self.save_path = save_path
        self.frame_counter = 0
        self.start_time = time.time()
        self.fps_counter = 0
        self.last_fps_time = time.time()

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
        return self.start_time

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

            if not self.acq_queue.empty():
                img_np, _ = self.acq_queue.get()
                # Convert to bytes for SDK input
                self.last_frame_data = img_np.tobytes()
    
            stInput.pData = cast(c_char_p(self.last_frame_data), POINTER(c_ubyte))
            stInput.nDataLen = len(self.last_frame_data)
            self.cam.MV_CC_InputOneFrame(stInput)
            self.frame_counter += 1

            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
    
            target_interval = self.frame_counter * target_interval_t - (time.time() - start_time)
            if target_interval > 0:
                time.sleep(target_interval)
            record_loop_end = time.time()
            self.recDuration.append(record_loop_end - record_loop_start)

    def stop_record(self):
        self.running = False
        self.recording = False
        self.mark_ttl_off()
        self.end_time = time.time()
        self.cam.MV_CC_StopRecord()
        return self.end_time
    
    def mark_ttl_on(self):
        self._send_to_arduino('T')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
    def trigger_laser(self):
        self._send_to_arduio('L')
    
    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

class OpenCVCamera:
    def __init__(self, root, arduino=None, camera_id=0):
        self.root = root
        self.arduino = arduino
        self.camera_id = camera_id
        self.display_name = f"Camera {camera_id+1} (OpenCV)"
        
        self.capture = None
        self.video_writer = None
        self.previewing = False
        self.recording = False
        self.frame_counter = 0
        self.start_time = None
        self.end_time = None
        self.current_fps = 0.0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        self.width = 1280
        self.height = 720
        self.fps = 30.0

    def open(self, index):
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        print(f"[INFO] Opened OpenCV camera {self.camera_id} with resolution {self.width}x{self.height} @ {self.fps:.2f}fps")

    def close(self):
        if self.capture:
            self.capture.release()
        if self.video_writer:
            self.video_writer.release()
        self.capture = None
        self.video_writer = None
        print(f"[INFO] OpenCV camera {self.camera_id} closed.")

    def set_parameters(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps

    def start_preview(self, callback):
        if self.previewing:
            print(f"[WARN] Camera {self.camera_id} preview already running.")
            return
        self.previewing = True
        self.preview_callback = callback
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def stop_preview(self):
        self.previewing = False

    def _preview_loop(self):
        while self.previewing:
            ret, frame = self.capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
                if self.preview_callback:
                    self.root.after(0, lambda img=pil_img: self.preview_callback(img, self.camera_id))
            else:
                time.sleep(0.01)

    def start_record(self, save_path):
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError("Camera not opened")
            
        self.recording = True
        self.save_path = save_path
        self.frame_counter = 0
        self.start_time = time.time()
        self.fps_counter = 0
        self.last_fps_time = time.time()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.video_writer.isOpened():
            raise RuntimeError("Could not open video writer")
            
        self.mark_ttl_on()
        threading.Thread(target=self._record_loop, daemon=True).start()
        return self.start_time

    def _record_loop(self):
        while self.recording:
            ret, frame = self.capture.read()
            if ret:
                self.video_writer.write(frame)

                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()

                target_frame_time = 1.0 / self.fps
                time.sleep(target_frame_time)
            else:
                time.sleep(0.01)

    def stop_record(self):
        self.recording = False
        self.mark_ttl_off()
        self.end_time = time.time()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print(f"Recording stopped for camera {self.camera_id}")
        return self.end_time
    
    def mark_ttl_on(self):
        self._send_to_arduino('T')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
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
        self.root.geometry("1400x900")

        self.arduino_port = "COM10"
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            print("Arduino port opened.")
            time.sleep(2)
        except serial.SerialException as e:
            messagebox.showerror("Arduino Error", f"Failed to open {self.arduino_port}: {e}, if you don't need to control fiber photometry, please ignore it!")
            self.arduino = None

        self.cameras = []
        self.camera_views = []
        
        self.timeline = []
        self.save_dir = os.getcwd()
        self.video_start_time = None
        self.event_log = []
        self.timestamp_log = []

        self.exposure_time = tk.DoubleVar(value=1000.0)
        self.resolution_choice = tk.StringVar(value="1280x1024")
        self.record_rate = tk.DoubleVar(value=30.0)
        self.sample_rate = tk.DoubleVar(value=90.0)

        self.selected_ai_channel = tk.StringVar(value="ai0")

        self.pixel_type = PixelType_Mono8
        self.selected_audio_device = None
        self.user_filename_prefix = tk.StringVar(value="experiment")

        self.enable_speed_sensor = tk.BooleanVar(value=False)
        self.speed_sensor = SpeedSensor(root)

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        config_frame = ttk.LabelFrame(left_panel, text="Block Configuration")
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(config_frame, text="Block Type:").grid(row=1, column=0, sticky="w", pady=2)
        self.block_type = tk.StringVar(value="Wait")
        type_combo = ttk.Combobox(config_frame, textvariable=self.block_type, 
                                 values=["Wait", "Sound", "Sound+Shock"], width=16)
        type_combo.grid(row=1, column=1, sticky="ew", pady=2)

        self.duration_frame = ttk.Frame(config_frame)
        self.duration_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        
        ttk.Label(self.duration_frame, text="Duration (s):").pack(side=tk.LEFT)
        self.block_duration = tk.StringVar(value="5")
        ttk.Entry(self.duration_frame, textvariable=self.block_duration, width=20).pack(side=tk.LEFT, padx=5)

        self.shock_frame = ttk.Frame(config_frame)
        
        ttk.Label(self.shock_frame, text="Sound Duration (s):").grid(row=0, column=0, sticky="w")
        self.sound_duration = tk.StringVar(value="5")
        ttk.Entry(self.shock_frame, textvariable=self.sound_duration, width=14).grid(row=0, column=1)
        
        ttk.Label(self.shock_frame, text="Shock Duration (s):").grid(row=1, column=0, sticky="w")
        self.shock_lead = tk.StringVar(value="2")
        ttk.Entry(self.shock_frame, textvariable=self.shock_lead, width=14).grid(row=1, column=1)

        # Button Frame
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        row1 = ttk.Frame(btn_frame)
        row1.pack(side=tk.TOP, pady=2)
        ttk.Button(row1, text="Add Block", command=self.add_block, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Remove Block", command=self.remove_selected_block, width=12).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(btn_frame)
        row2.pack(side=tk.TOP, pady=2)

        ttk.Button(row2, text="Clear All", command=self.clear_blocks, width=26).pack(side=tk.TOP)

        file_frame = ttk.LabelFrame(left_panel, text="File Settings")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Filename Prefix:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(file_frame, textvariable=self.user_filename_prefix, width=17).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Choose Save Directory", command=self.choose_directory, width=30).grid(row=1, column=0, columnspan=2, pady=5)

        device_frame = ttk.LabelFrame(left_panel, text="Device Control")
        device_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(device_frame, text="Add Camera", command=self.add_camera, width=14).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="Remove Camera", command=self.remove_camera, width=14).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(device_frame, text="Edit Camera", command=self.edit_camera, width=14).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="Clear All Cameras", command=self.clear_all_cameras, width=14).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(device_frame, text="Select Speaker", command=self.select_devices, width=14).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="Test Speaker", command=self.test_speaker, width=14).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(device_frame, text="Start Preview All", command=self.start_preview_all, width=14).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="Stop Preview All", command=self.stop_preview_all, width=14).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(device_frame, text="START", command=self.run_timeline, 
                  style="Accent.TButton", width=14).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(device_frame, text="EXIT", command=self.exit, 
                  style="Accent.TButton", width=14).grid(row=3, column=1, padx=5, pady=5)

        speed_frame = ttk.LabelFrame(left_panel, text="Speed Sensor Control")
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(speed_frame, text="Enable Speed Sensor", variable=self.enable_speed_sensor,
                       command=self.toggle_speed_sensor).grid(row=0, column=0, padx=5, pady=5)

        self.speed_settings_frame = ttk.Frame(speed_frame)
        self.speed_settings_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.speed_settings_frame, text="Input Rate (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.speed_input_rate = tk.IntVar(value=1000)
        ttk.Entry(self.speed_settings_frame, textvariable=self.speed_input_rate, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.speed_settings_frame, text="Input Channels:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        dev_menu = ttk.OptionMenu(self.speed_settings_frame, self.selected_ai_channel, "ai0", "ai0", "ai1", "ai2", "ai3")
        dev_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.toggle_speed_sensor()

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_panel_top = ttk.Frame(right_panel)
        right_panel_top.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        right_panel_top.configure(width=960) 
        right_panel_top.configure(height=300) 

        # ---------------- Timeline Sequence ----------------
        timeline_frame = ttk.LabelFrame(right_panel_top, text="Timeline Sequence")
        timeline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Timeline Listbox + Scrollbars
        timeline_listbox_frame = ttk.Frame(timeline_frame)
        timeline_listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.timeline_listbox = tk.Listbox(timeline_listbox_frame, font=("Arial", 10), height=5)
        timeline_scrollbar_y = ttk.Scrollbar(timeline_listbox_frame, orient="vertical", command=self.timeline_listbox.yview)
        timeline_scrollbar_x = ttk.Scrollbar(timeline_listbox_frame, orient="horizontal", command=self.timeline_listbox.xview)

        self.timeline_listbox.configure(yscrollcommand=timeline_scrollbar_y.set, xscrollcommand=timeline_scrollbar_x.set)

        self.timeline_listbox.grid(row=0, column=0, sticky="nsew")
        timeline_scrollbar_y.grid(row=0, column=1, sticky="ns")
        timeline_scrollbar_x.grid(row=1, column=0, sticky="ew")

        timeline_listbox_frame.grid_rowconfigure(0, weight=1)
        timeline_listbox_frame.grid_columnconfigure(0, weight=1)

        self.timeline_listbox.bind("<<ListboxSelect>>", self.sync_selection)

        # ---------------- Camera Management ----------------
        camera_mgmt_frame = ttk.LabelFrame(right_panel_top, text="Camera Management")
        camera_mgmt_frame.grid(row=0, column=1, sticky="nsew")

        # Camera Listbox + Scrollbars
        camera_listbox_frame = ttk.Frame(camera_mgmt_frame)
        camera_listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.camera_listbox = tk.Listbox(camera_listbox_frame, height=4)
        camera_scrollbar_y = ttk.Scrollbar(camera_listbox_frame, orient="vertical", command=self.camera_listbox.yview)
        camera_scrollbar_x = ttk.Scrollbar(camera_listbox_frame, orient="horizontal", command=self.camera_listbox.xview)

        self.camera_listbox.configure(yscrollcommand=camera_scrollbar_y.set, xscrollcommand=camera_scrollbar_x.set)

        self.camera_listbox.grid(row=0, column=0, sticky="nsew")
        camera_scrollbar_y.grid(row=0, column=1, sticky="ns")
        camera_scrollbar_x.grid(row=1, column=0, sticky="ew")

        camera_listbox_frame.grid_rowconfigure(0, weight=1)
        camera_listbox_frame.grid_columnconfigure(0, weight=1)

        right_panel_top.grid_columnconfigure(0, weight=1)
        right_panel_top.grid_columnconfigure(1, weight=1)
        right_panel_top.grid_rowconfigure(0, weight=1)

        self.video_container = ttk.LabelFrame(right_panel, text="Camera Previews")
        self.video_container.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        self.video_container.configure(width=960) 
        self.video_container.configure(height=400) 
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(1, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(1, weight=1)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.recording_label = ttk.Label(status_frame, text="Status: Ready")
        self.recording_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(status_frame, text="00:00:00")
        self.time_label.pack(side=tk.LEFT, padx=20)

        self.root.style = ttk.Style()
        self.root.style.configure("Accent.TButton", font=("Arial", 10, "bold"), foreground="black", background="#4CAF50")

        type_combo.bind("<<ComboboxSelected>>", self.on_type_change)
        
    def toggle_speed_sensor(self):
        """启用或禁用速度传感器设置区域"""
        state = tk.NORMAL if self.enable_speed_sensor.get() else tk.DISABLED
        for child in self.speed_settings_frame.winfo_children():
            child.configure(state=state)

    def on_type_change(self, event=None):
        # Adjust visible input fields based on selected block type
        btype = self.block_type.get()
        if btype == "Sound+Shock":
            # Hide single duration field
            self.duration_frame.grid_remove()
            # Show Sound+Shock fields
            self.shock_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        else:
            # Hide Sound+Shock specific fields
            self.shock_frame.grid_remove()
            # Show single duration field
            self.duration_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

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

    def add_camera(self):
        popup = tk.Toplevel(self.root)
        popup.title("Add Camera")
        popup.resizable(False, False)
        popup.grab_set()
        
        ttk.Label(popup, text="Camera Type:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        camera_type = tk.StringVar(value="HKCamera")
        camera_type_combo = ttk.Combobox(popup, textvariable=camera_type, 
                                       values=["HKCamera", "OpenCVCamera"], width=15)
        camera_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Create frames for different camera types
        hk_frame = ttk.LabelFrame(popup, text="HIK Camera Parameters")
        hk_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        opencv_frame = ttk.LabelFrame(popup, text="OpenCV Camera Parameters")
        opencv_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        common_frame = ttk.Frame(popup)
        common_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # HIK Camera specific parameters
        ttk.Label(hk_frame, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        exposure_var = tk.DoubleVar(value=self.exposure_time.get())
        ttk.Entry(hk_frame, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(hk_frame, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        sample_rate_var = tk.DoubleVar(value=self.sample_rate.get())
        ttk.Entry(hk_frame, textvariable=sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # OpenCV Camera specific parameters
        camera_list = list_available_cameras()
        ttk.Label(opencv_frame, text="Camera Device:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        camera_var = tk.StringVar()
        camera_combo = ttk.Combobox(opencv_frame, textvariable=camera_var, values=camera_list, width=30)
        camera_combo.grid(row=0, column=1, padx=5, pady=5)
        if camera_list:
            camera_combo.current(0)

        # Common parameters for both camera types
        ttk.Label(common_frame, text="Resolution:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        resolution_var = tk.StringVar(value=self.resolution_choice.get())
        resolution_menu = ttk.OptionMenu(common_frame, resolution_var, resolution_var.get(), 
                                       "1920x1080","1280x1024", "1280x720", "640x480")
        resolution_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(common_frame, text="Recording FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        record_rate_var = tk.DoubleVar(value=self.record_rate.get())
        ttk.Entry(common_frame, textvariable=record_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # Function to toggle parameter visibility
        def toggle_parameters():
            if camera_type.get() == "HKCamera":
                hk_frame.grid()
                opencv_frame.grid_remove()
            else:
                hk_frame.grid_remove()
                opencv_frame.grid()
        
        camera_type_combo.bind("<<ComboboxSelected>>", lambda e: toggle_parameters())
        toggle_parameters()  # Initialize visibility

        def create_camera():
            try:
                camera_id = len(self.cameras)

                res_text = resolution_var.get()
                width, height = map(int, res_text.split("x"))

                if camera_type.get() == "HKCamera":
                    camera = HKCamera(self.root, self.arduino, camera_id=camera_id)
                    camera.set_parameters(
                        width=width,
                        height=height,
                        exposure_time=exposure_var.get(),
                        sample_rate=sample_rate_var.get(),
                        record_rate=record_rate_var.get()
                    )
                    camera.open()
                else:
                    camera = OpenCVCamera(self.root, self.arduino, camera_id=camera_id)
                    camera.set_parameters(
                        width=width,
                        height=height,
                        fps=record_rate_var.get()
                    )
                    selected_camera_name = camera_var.get()
                    camera_index = camera_list.index(selected_camera_name)
                    camera.open(camera_index)
                
                self.cameras.append(camera)
                self.create_camera_view(camera_id)
                self.update_camera_list()
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Camera Error", f"Failed to add camera: {e}")

        btn_frame = ttk.Frame(popup)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Add Camera", command=create_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=popup.destroy).pack(side=tk.LEFT, padx=5)

    def remove_camera(self):
        """Remove the selected camera"""
        sel = self.camera_listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select a camera to remove")
            return
            
        index = sel[0]
        camera = self.cameras[index]

        camera.stop_preview()
        camera.stop_record()
        camera.close()

        del self.cameras[index]

        self.update_camera_list()
        self.update_camera_views()
        messagebox.showinfo("Info", f"Camera {index} removed successfully")

    def edit_camera(self):
        """Edit the parameters of selected camera"""
        sel = self.camera_listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select a camera to edit")
            return
            
        index = sel[0]
        camera = self.cameras[index]

        popup = tk.Toplevel(self.root)
        popup.title(f"Edit Camera {index}")
        popup.resizable(False, False)
        popup.grab_set()

        if isinstance(camera, HKCamera):
            ttk.Label(popup, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5)
            exposure_var = tk.DoubleVar(value=camera.exposure_time)
            ttk.Entry(popup, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(popup, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5)
            sample_rate_var = tk.DoubleVar(value=camera.sample_rate)
            ttk.Entry(popup, textvariable=sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(popup, text="Recording FPS:").grid(row=2, column=0, padx=5, pady=5)
            record_rate_var = tk.DoubleVar(value=camera.record_rate)
            ttk.Entry(popup, textvariable=record_rate_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        else:
            ttk.Label(popup, text="Recording FPS:").grid(row=0, column=0, padx=5, pady=5)
            record_rate_var = tk.DoubleVar(value=camera.fps)
            ttk.Entry(popup, textvariable=record_rate_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        def apply_changes():
            if isinstance(camera, HKCamera):
                camera.exposure_time = exposure_var.get()
                camera.sample_rate = sample_rate_var.get()
                camera.record_rate = record_rate_var.get()
                try:
                    camera.cam.MV_CC_SetFloatValue("ExposureTime", camera.exposure_time)
                    camera.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", camera.sample_rate)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to apply parameters: {e}")
            else:
                camera.fps = record_rate_var.get()
                if camera.capture and camera.capture.isOpened():
                    camera.capture.set(cv2.CAP_PROP_FPS, camera.fps)
            
            popup.destroy()
            messagebox.showinfo("Info", "Camera parameters updated")
        
        ttk.Button(popup, text="Apply", command=apply_changes).grid(row=3, column=0, columnspan=2, pady=10)

    def clear_all_cameras(self):
        if not self.cameras:
            messagebox.showinfo("Info", "No cameras to clear")
            return

        if not messagebox.askyesno("Confirm", "Are you sure you want to remove all cameras?"):
            return

        for camera in self.cameras:
            camera.stop_preview()
            camera.stop_record()
            camera.close()

        self.cameras = []

        for view in self.camera_views:
            view["frame"].destroy()
        self.camera_views = []

        self.update_camera_list()

        messagebox.showinfo("Info", "All cameras have been removed")

    def update_camera_list(self):
        """更新相机列表显示"""
        self.camera_listbox.delete(0, tk.END)
        for i, camera in enumerate(self.cameras):
            self.camera_listbox.insert(tk.END, f"Camera {i}: {camera.display_name}")

    def create_camera_view(self, camera_id):
        """Create Preview for Camera"""
        camera = self.cameras[camera_id]

        num_views = len(self.camera_views)
        row = num_views // 2
        col = num_views % 2

        camera_frame = ttk.LabelFrame(self.video_container, text=camera.display_name)
        camera_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        camera_frame.grid_rowconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(0, weight=1)

        video_label = ttk.Label(camera_frame)
        video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.camera_views.append({
            "frame": camera_frame,
            "label": video_label,
            "camera_id": camera_id
        })

        self.video_container.grid_rowconfigure(row, weight=1)
        self.video_container.grid_columnconfigure(col, weight=1)

    def update_camera_views(self):
        """更新相机视图布局"""
        for view in self.camera_views:
            view["frame"].destroy()
        self.camera_views = []

        for camera_id in range(len(self.cameras)):
            self.create_camera_view(camera_id)

    def update_image(self, pil_image, camera_id):
        """Ungrade Unique Camera Preview"""
        for view in self.camera_views:
            if view["camera_id"] == camera_id:
                target_size = (480, 360)
                padded = ImageOps.pad(pil_image, target_size, color=0, centering=(0.5, 0.5))
                imgtk = ImageTk.PhotoImage(padded)
                view["label"].imgtk = imgtk
                view["label"].configure(image=imgtk)
                camera = self.cameras[camera_id]
                view["frame"].config(text=f"{camera.display_name} - FPS: {camera.current_fps:.1f}")
                break

    def start_preview_all(self):
        """Start Preview All Camera"""
        for camera in self.cameras:
            try:
                camera.start_preview(self.update_image)
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to start preview: {str(e)}")
        self.recording_label.config(text="Status: Previewing All Cameras")

    def stop_preview_all(self):
        """Stop Preview All Camera"""
        for camera in self.cameras:
            camera.stop_preview()
        self.recording_label.config(text="Status: Preview Stopped")

    def update_timer(self):
        if any(camera.recording for camera in self.cameras):
            elapsed = time.time() - self.video_start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            self.time_label.config(text=f"{hours:02d}:{mins:02d}:{secs:02d}")
            self.root.after(1000, self.update_timer)

    def run_timeline(self):
        if not self.timeline:
            messagebox.showerror("Error", "Timeline is empty")
            return
        if not self.cameras:
            messagebox.showerror("Error", "No cameras added")
            return
            
        threading.Thread(target=self._run_timeline, daemon=True).start()

    def _run_timeline(self):
        self.recording_label.config(text="Status: Running Experiment")
        self.event_log = []
        self.timestamp_log = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = self.user_filename_prefix.get()

        speed_sensor_enabled = False
        speed_start = speed_end = 0
        if self.enable_speed_sensor.get():
            try:
                self.speed_sensor.input_rate = self.speed_input_rate.get()
                self.speed_sensor.channels = [ch.strip() for ch in self.speed_channels.get().split(',')]

                speed_file = os.path.join(self.save_dir, f"{prefix}_speed_{timestamp}.bin")
                if self.speed_sensor.start_tracking(speed_file):
                    speed_sensor_enabled = True
                    speed_start = time.time()
                    print(f"Speed sensor recording started: {speed_file}")
                    self.timestamp_log.append(("Speed Sensor", "Start", speed_start))
            except Exception as e:
                messagebox.showerror("Speed Sensor Error", f"Failed to start speed sensor: {e}")

        camera_timestamps = []
        for camera in self.cameras:
            filename = f"{prefix}_cam{camera.camera_id}_{timestamp}.avi"
            save_path = os.path.join(self.save_dir, filename)
            try:
                start_time = camera.start_record(save_path)
                camera_timestamps.append((camera.camera_id, start_time))
                self.timestamp_log.append((f"Camera {camera.camera_id}", "Start", start_time))
            except Exception as e:
                messagebox.showerror("Recording Error", f"Failed to start recording for camera {camera.camera_id}: {e}")
        
        self.video_start_time = time.time()
        self.update_timer()

        experiment_start = time.time()
        self.timestamp_log.append(("Experiment", "Start", experiment_start))

        i = 0
        while i < len(self.timeline) and any(camera.recording for camera in self.cameras):
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

        for camera in self.cameras:
            if camera.recording:
                end_time = camera.stop_record()
                self.timestamp_log.append((f"Camera {camera.camera_id}", "End", end_time))

        if speed_sensor_enabled:
            start_time, end_time = self.speed_sensor.stop_tracking()
            self.timestamp_log.append(("Speed Sensor", "End", end_time))
            print("Speed sensor recording stopped")

        experiment_end = time.time()
        self.timestamp_log.append(("Experiment", "End", experiment_end))

        self.save_logs(prefix, timestamp)
        self.recording_label.config(text="Status: Experiment Complete")
        print("Experiment complete")
        
    def save_logs(self, prefix, timestamp):
        """保存事件日志和时间戳日志"""
        event_file = os.path.join(self.save_dir, f"{prefix}_events_{timestamp}.csv")
        try:
            with open(event_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["start_time", "end_time", "Event Type"])
                for start, end, event in self.event_log:
                    writer.writerow([f"{start:.3f}", f"{end:.3f}", event])
            print(f"Event log saved: {event_file}")
        except Exception as e:
            print(f"Failed to save event log: {e}")

        timestamp_file = os.path.join(self.save_dir, f"{prefix}_timestamps_{timestamp}.csv")
        try:
            with open(timestamp_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Device", "Action", "Timestamp"])
                for device, action, timestamp in self.timestamp_log:
                    writer.writerow([device, action, f"{timestamp:.6f}"])
            print(f"Timestamp log saved: {timestamp_file}")
        except Exception as e:
            print(f"Failed to save timestamp log: {e}")

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

    def on_closing(self):
        for camera in self.cameras:
            camera.stop_preview()
            camera.stop_record()
            camera.close()

        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")

        if hasattr(self, 'speed_sensor') and self.speed_sensor.running:
            self.speed_sensor.stop_tracking()

        self.root.destroy()
    
    def exit(self):
        for camera in self.cameras:
            camera.stop_preview()
            camera.close()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")

        if hasattr(self, 'speed_sensor') and self.speed_sensor.running:
            self.speed_sensor.stop_tracking()
            
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FearTrainingGUI(root)
    root.mainloop()