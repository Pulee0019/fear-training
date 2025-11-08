# -*- coding: utf-8 -*-
"""
Created on Tue Jul 8 19:18:40 2025

@author: Pulee

Exit Error
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
from functools import partial
import matplotlib
matplotlib.use('TkAgg')

PixelType_Mono8 = 17301505

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

class SpeedEncoderGUI:
    """集成速度传感器GUI功能"""
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.running = False
        self.task = None
        self.file = None
        
        # 配置参数
        self.input_rate = 1000
        self.down_sample = 50
        self.speed_down_sample_factor = 50
        self.device = 'Dev1'
        self.channels = ['Dev1/ai0', 'Dev1/ai1', 'Dev1/ai2', 'Dev1/ai3']
        self.num_channels = len(self.channels)
        self.scale = 2 ** 15 / 10
        
        # 数据结构
        self.buffer_size = 50000  # 预分配缓冲区大小
        self.data = np.zeros((self.num_channels, self.buffer_size))
        self.time = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.voltage_range = np.array([[0, 5] for _ in range(self.num_channels)])
        self.speed_data = [[] for _ in range(self.num_channels)]
        self.transition_counts = np.zeros(self.num_channels)
        self._speed_accumulator = [[] for _ in range(self.num_channels)]
        
        # 状态变量
        self.data_updated = False
        
        # UI状态变量
        self.enabled_channels = [tk.BooleanVar(value=True) for _ in range(self.num_channels)]
        self.view_mode = [tk.BooleanVar(value=False) for _ in range(self.num_channels)]
        self.auto_scale = [tk.BooleanVar(value=True) for _ in range(self.num_channels)]
        self.x_range = [tk.StringVar(value='') for _ in range(self.num_channels)]
        self.y_range = [tk.StringVar(value='') for _ in range(self.num_channels)]
        
        # 初始化UI
        self.init_ui()
        
        # 启动更新循环
        self.active = True
        self.update_display_loop()
        
    def init_ui(self):
        """初始化速度传感器UI"""
        # 通道控制框架
        ch_frame = ttk.LabelFrame(self.parent, text="Channel Controls")
        ch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 为每个通道创建控制行
        for i in range(self.num_channels):
            row_frame = ttk.Frame(ch_frame)
            row_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # 通道启用复选框
            chk = ttk.Checkbutton(row_frame, text=f"Ch{i}", variable=self.enabled_channels[i])
            chk.pack(side=tk.LEFT, padx=5)
            
            # 显示速度复选框
            speed_chk = ttk.Checkbutton(row_frame, text="Show Speed", variable=self.view_mode[i])
            speed_chk.pack(side=tk.LEFT, padx=5)
            
            # 自动缩放复选框
            auto_chk = ttk.Checkbutton(row_frame, text="Auto Y", variable=self.auto_scale[i])
            auto_chk.pack(side=tk.LEFT, padx=5)
            
            # X轴范围
            ttk.Label(row_frame, text="X Lim:").pack(side=tk.LEFT, padx=(10,0))
            x_entry = ttk.Entry(row_frame, textvariable=self.x_range[i], width=10)
            x_entry.pack(side=tk.LEFT)
            
            # Y轴范围
            ttk.Label(row_frame, text="Y Lim:").pack(side=tk.LEFT, padx=(10,0))
            y_entry = ttk.Entry(row_frame, textvariable=self.y_range[i], width=10)
            y_entry.pack(side=tk.LEFT)
        
        # 控制按钮框架
        btn_frame = ttk.Frame(self.parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 应用轴限制按钮
        apply_btn = ttk.Button(btn_frame, text="Apply Limits", command=self.apply_axis_limits)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        # 自动校准按钮
        calibrate_btn = ttk.Button(btn_frame, text="Auto Calibrate", command=self.auto_calibrate_voltage_range)
        calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        # 导出按钮
        export_frame = ttk.Frame(btn_frame)
        export_frame.pack(side=tk.RIGHT, padx=5)
        
        csv_btn = ttk.Button(export_frame, text="Export CSV", command=self.export_csv)
        csv_btn.pack(side=tk.LEFT, padx=5)
        
        mat_btn = ttk.Button(export_frame, text="Export MAT", command=self.export_mat)
        mat_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_var = tk.StringVar(value="Speed sensor ready")
        status_label = ttk.Label(self.parent, textvariable=self.status_var)
        status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建图表
        self.create_speed_plot()
        
    def create_speed_plot(self):
        """创建速度图表"""
        # 创建图表框架
        self.plot_frame = ttk.LabelFrame(self.parent, text="Speed Plots")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(self.num_channels, 1, figsize=(8, 6), sharex=True)
        if self.num_channels == 1:
            self.ax = [self.ax]  # 确保ax总是列表
        
        # 初始化图表
        self.lines = []
        for i, ax in enumerate(self.ax):
            line, = ax.plot([], [], label=f"Channel {i}")
            self.lines.append(line)
            ax.set_xlim(0, 10)
            ax.set_ylim(-10, 10)
            ax.grid(False)
            ax.legend(loc='upper right')
            ax.set_ylabel("Voltage (V)" if not self.view_mode[i].get() else "Speed (°/s)")
        
        self.fig.tight_layout()
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始时隐藏图表
        self.plot_frame.pack_forget()
        
    def start_tracking(self, file_path):
        """开始数据采集"""
        if self.running:
            return False
            
        try:
            self.file = open(file_path, 'wb')
        except IOError as e:
            messagebox.showerror("Error", f"Could not open file: {e}")
            return False
            
        try:
            self.task = Task()
            # 只添加启用的通道
            enabled_count = 0
            for i, ch in enumerate(self.channels):
                if self.enabled_channels[i].get():
                    self.task.CreateAIVoltageChan(
                        ch, "", DAQmx_Val_Cfg_Default, 
                        -10.0, 10.0, DAQmx_Val_Volts, None
                    )
                    enabled_count += 1
                    
            if enabled_count == 0:
                raise ValueError("No channels enabled")
                
            # 配置定时和回调
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
            
            # 重置数据结构
            self._reset_data()
            
            return True
        except Exception as e:
            messagebox.showerror("DAQ Error", f"Failed to start acquisition: {e}")
            if self.file:
                self.file.close()
                self.file = None
            return False

    def _reset_data(self):
        """重置所有数据结构"""
        self.current_idx = 0
        self.data = np.zeros((self.num_channels, self.buffer_size))
        self.time = np.zeros(self.buffer_size)
        self.speed_data = [[] for _ in range(self.num_channels)]
        self.transition_counts = np.zeros(self.num_channels)
        self._speed_accumulator = [[] for _ in range(self.num_channels)]
        
    def listener_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        """DAQmx数据采集回调函数"""
        try:
            read = ctypes.c_int32()
            # 只为启用的通道分配缓冲区
            active_channels = sum(1 for ch in self.enabled_channels if ch.get())
            buffer = np.zeros((active_channels, number_of_samples))
            
            # 从DAQ读取数据
            self.task.ReadAnalogF64(
                number_of_samples, 10.0, DAQmx_Val_GroupByChannel,
                buffer, buffer.size, ctypes.byref(read), None
            )
            samples = read.value
            
            self.parent.after(0, lambda: self.handle_data(buffer.copy(), samples))
            return 0
        except Exception as e:
            print(f"Error in callback: {e}")
            return -1

    def handle_data(self, buffer, samples):
        """处理采集到的数据"""
        # 检查是否需要调整数据缓冲区大小
        if self.current_idx + samples > self.data.shape[1]:
            new_size = max(self.data.shape[1] * 2, self.current_idx + samples)
            self.data = np.pad(self.data, ((0, 0), (0, new_size - self.data.shape[1])), mode='constant')
            self.time = np.pad(self.time, (0, new_size - len(self.time)), mode='constant')
            
        # 更新时间数组
        start_time = self.time[self.current_idx - 1] + 1 / self.input_rate if self.current_idx else 0
        self.time[self.current_idx:self.current_idx + samples] = np.linspace(
            start_time, start_time + samples / self.input_rate, samples)
        
        # 将缓冲区数据映射到启用的通道
        buffer_idx = 0
        for i in range(self.num_channels):
            if self.enabled_channels[i].get():
                self.data[i, self.current_idx:self.current_idx + samples] = buffer[buffer_idx, :samples]
                buffer_idx += 1
        
        # 更新速度计算
        self.update_speed_data(samples)
        
        # 保存二进制数据
        if self.file and not self.file.closed:
            self.save_data_binary(buffer[:, :samples])
            
        self.current_idx += samples
        self.data_updated = True

    def update_speed_data(self, samples):
        """处理数据计算速度信息"""
        for i in range(self.num_channels):
            if not self.enabled_channels[i].get():
                continue
                
            # 获取最新的数据段
            start_idx = max(0, self.current_idx - samples)
            ch_data = self.data[i, start_idx:self.current_idx]
            
            # 添加到累加器
            self._speed_accumulator[i].extend(ch_data.tolist())
            
            # 处理完整的数据段
            while len(self._speed_accumulator[i]) >= self.speed_down_sample_factor:
                seg = self._speed_accumulator[i][:self.speed_down_sample_factor]
                self._speed_accumulator[i] = self._speed_accumulator[i][self.speed_down_sample_factor:]
                
                # 计算此段的时间值
                t = np.arange(len(seg)) / self.input_rate
                
                # 计算速度和过渡计数
                speed, transition = self.compute_speed(np.array(seg), t, self.voltage_range[i])
                self.speed_data[i].append(speed)
                self.transition_counts[i] += transition

    def compute_speed(self, data, time, voltage_range):
        """从信号数据计算旋转速度"""
        is_transition = 0
        delta_voltage = voltage_range[1] - voltage_range[0]
        
        # 安全检查零增量
        if abs(delta_voltage) < 1e-6:
            return 0.0, 0
            
        # 计算过渡阈值
        thresh = 3 / 5 * delta_voltage
        
        # 查找信号中的过渡
        diff_data = np.diff(data)
        ind = np.where(np.abs(diff_data) > 3)[0]
        
        # 处理过渡
        for i in ind:
            is_transition = 1
            if diff_data[i] < thresh:
                data[i + 1:] += delta_voltage
            elif diff_data[i] > thresh:
                data[i + 1:] -= delta_voltage
        
        # 转换为角度
        data_deg = (data / delta_voltage) * 360
        
        # 计算平均角度变化
        if len(data_deg) >= 20:  # 确保有足够的数据点
            delta_deg = np.mean(data_deg[-10:]) - np.mean(data_deg[:10])
            
            # 处理环绕
            if delta_deg > 200: 
                delta_deg -= 360
            elif delta_deg < -200: 
                delta_deg += 360
                
            # 计算持续时间和速度
            duration = np.mean(time[-10:]) - np.mean(time[:10])
            speed = delta_deg / duration if duration > 1e-6 else 0.0
        else:
            speed = 0.0
            
        return speed, is_transition

    def save_data_binary(self, data_chunk):
        """以二进制格式保存数据"""
        scaled = (data_chunk * self.scale).astype(np.int16)
        self.file.write(scaled.tobytes())
        
    def stop_tracking(self):
        """停止数据采集"""
        if not self.running: 
            return
            
        # 停止并清除任务
        if self.task:
            try:
                self.task.StopTask()
                self.task.ClearTask()
            except Exception as e:
                print(f"Error stopping task: {e}")
            finally:
                self.task = None
                
        # 关闭输出文件
        if hasattr(self, 'file') and self.file and not self.file.closed:
            self.file.close()
            self.file = None
            
        self.running = False
        
    def update_display_loop(self):
        """定期更新显示"""
        if not self.active:
            return
            
        if self.data_updated:
            self.update_display()
            self.data_updated = False

            if self.active:
                self.root.after(200, self.update_display_loop)

    def update_display(self):
        """更新图表显示"""
        if self.current_idx < self.speed_down_sample_factor:
            return
            
        for i in range(self.num_channels):
            if not self.enabled_channels[i].get():
                continue
                
            view_speed = self.view_mode[i].get()
            if view_speed and self.speed_data[i]:
                # 显示速度数据
                y_data = np.array(self.speed_data[i])
                x_data = np.linspace(0, len(y_data) / (self.input_rate / self.speed_down_sample_factor), len(y_data))
                self.ax[i].set_ylabel("Speed (°/s)")
            else:
                # 显示原始电压数据
                end_idx = min(self.current_idx, len(self.time))
                y_data = self.data[i, :end_idx:self.down_sample]
                x_data = self.time[:end_idx:self.down_sample]
                y_data = y_data[:len(x_data)]  # 确保长度匹配
                self.ax[i].set_ylabel("Voltage (V)")
                
            self.lines[i].set_data(x_data, y_data)
            
            # 设置坐标轴范围
            if len(x_data) > 0:
                self.ax[i].set_xlim(0, x_data[-1])
                
            if self.auto_scale[i].get():
                self.ax[i].set_autoscale_on(True)
                self.ax[i].relim()
                self.ax[i].autoscale_view()
            else:
                try:
                    xlim_str = self.x_range[i].get().strip()
                    ylim_str = self.y_range[i].get().strip()
                    
                    if xlim_str:
                        xlim = list(map(float, xlim_str.split(',')))
                        if len(xlim) == 2:
                            self.ax[i].set_xlim(xlim)
                            
                    if ylim_str:
                        ylim = list(map(float, ylim_str.split(',')))
                        if len(ylim) == 2:
                            self.ax[i].set_ylim(ylim)
                except ValueError:
                    pass  # 忽略无效输入
        
        # 更新图表和状态栏
        self.canvas.draw_idle()
        self._update_status_bar()
        
    def _update_status_bar(self):
        """用当前速度和过渡信息更新状态栏"""
        avg_speeds = []
        for i in range(self.num_channels):
            if self.speed_data[i]:
                # 计算最后5个速度测量的平均值以获得更稳定的显示
                recent_speeds = self.speed_data[i][-5:] if len(self.speed_data[i]) >= 5 else self.speed_data[i]
                avg_speed = np.mean(recent_speeds)
            else:
                avg_speed = 0
            avg_speeds.append(avg_speed)
            
        transitions = self.transition_counts.astype(int)
        
        # 格式化状态信息
        status = " | ".join([
            f"Ch{i}: {avg_speeds[i]:.1f}°/s, {transitions[i]} trans." 
            for i in range(self.num_channels) if self.enabled_channels[i].get()
        ])
        self.status_var.set(status)

    def apply_axis_limits(self):
        """应用用户指定的坐标轴范围"""
        self.data_updated = True

    def auto_calibrate_voltage_range(self):
        """从信号自动校准电压范围"""
        if self.current_idx == 0:
            messagebox.showinfo("Info", "No data available for calibration.")
            return
            
        for i in range(self.num_channels):
            if self.enabled_channels[i].get():
                signal = self.data[i, :self.current_idx]
                
                # 查找信号中的显著跳跃
                diff = np.diff(signal)
                jump_indices = np.where(np.abs(diff) > 2.5)[0]
                
                if len(jump_indices) > 0:
                    v_min = np.min(signal)
                    v_max = np.max(signal)
                    
                    # 设置带小边界的电压范围
                    v_margin = (v_max - v_min) * 0.1  # 10% 边界
                    self.voltage_range[i] = [v_min - v_margin, v_max + v_margin]
                else:
                    messagebox.showinfo("Info", f"No significant transitions found in Channel {i}.")
                    
        messagebox.showinfo("Done", "Voltage range calibrated from signal.")

    def export_csv(self):
        """将速度数据导出到CSV文件"""
        if not any(self.speed_data):
            messagebox.showinfo("Info", "No speed data to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV files", "*.csv")]
        )
        if not filename: 
            return
            
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # 写入表头
                header = [f"Time (s)"] + [f"Ch{i} (°/s)" for i in range(self.num_channels) 
                                         if self.enabled_channels[i].get() and self.speed_data[i]]
                writer.writerow(header)
                
                # 获取通道中速度数据的最大长度
                max_len = max((len(data) for data in self.speed_data if data), default=0)
                
                # 创建时间列
                time_values = np.linspace(0, max_len / (self.input_rate / self.speed_down_sample_factor), max_len)
                
                # 写入数据行
                for idx in range(max_len):
                    row = [f"{time_values[idx]:.3f}"]
                    for i in range(self.num_channels):
                        if self.enabled_channels[i].get() and self.speed_data[i]:
                            value = self.speed_data[i][idx] if idx < len(self.speed_data[i]) else ""
                            row.append(value)
                    writer.writerow(row)
                    
            messagebox.showinfo("Export", "CSV export completed successfully.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV: {e}")
    
    def export_mat(self):
        """将速度数据导出到MATLAB .mat文件"""
        if not any(self.speed_data):
            messagebox.showinfo("Info", "No speed data to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".mat", 
            filetypes=[("MAT files", "*.mat")]
        )
        if not filename: 
            return
            
        try:
            # 为MATLAB导出创建字典
            mat_dict = {}
            
            # 添加时间向量
            max_len = max((len(data) for data in self.speed_data if data), default=0)
            if max_len > 0:
                mat_dict["time"] = np.linspace(0, max_len / (self.input_rate / self.speed_down_sample_factor), max_len)
            
            # 添加通道数据
            for i in range(self.num_channels):
                if self.enabled_channels[i].get() and self.speed_data[i]:
                    mat_dict[f"Ch{i}"] = np.array(self.speed_data[i])
            
            # 添加元数据
            mat_dict["metadata"] = {
                "sample_rate": self.input_rate,
                "down_sample_factor": self.speed_down_sample_factor,
                "acquisition_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            import scipy.io
            scipy.io.savemat(filename, mat_dict)
            messagebox.showinfo("Export", "MAT export completed successfully.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export MAT file: {e}")

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
        self.speed_encoder_gui = None  # 速度传感器GUI组件

        self.active_after_ids = []
        self.preview_active = False

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        # 速度传感器控制区域
        speed_frame = ttk.LabelFrame(left_panel, text="Speed Sensor Control")
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 启用速度传感器复选框
        ttk.Checkbutton(speed_frame, text="Enable Speed Sensor", variable=self.enable_speed_sensor,
                       command=self.toggle_speed_sensor).grid(row=0, column=0, padx=5, pady=5)
        
        # 速度传感器设置框架
        self.speed_settings_frame = ttk.Frame(speed_frame)
        self.speed_settings_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.speed_settings_frame, text="Input Rate (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.speed_input_rate = tk.IntVar(value=1000)
        ttk.Entry(self.speed_settings_frame, textvariable=self.speed_input_rate, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 初始时禁用速度传感器设置
        self.toggle_speed_sensor()
        
        # 实验配置区域
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

        # 按钮框架
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        row1 = ttk.Frame(btn_frame)
        row1.pack(side=tk.TOP, pady=2)
        ttk.Button(row1, text="Add Block", command=self.add_block, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Remove Block", command=self.remove_selected_block, width=12).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(btn_frame)
        row2.pack(side=tk.TOP, pady=2)

        ttk.Button(row2, text="Clear All", command=self.clear_blocks, width=26).pack(side=tk.TOP)

        # 文件设置区域
        file_frame = ttk.LabelFrame(left_panel, text="File Settings")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Filename Prefix:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(file_frame, textvariable=self.user_filename_prefix, width=17).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Choose Save Directory", command=self.choose_directory, width=30).grid(row=1, column=0, columnspan=2, pady=5)

        # 设备控制区域
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

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_panel_top = ttk.Frame(right_panel)
        right_panel_top.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        right_panel_top.configure(width=960) 
        right_panel_top.configure(height=300) 

        # ---------------- 时间线序列 ----------------
        timeline_frame = ttk.LabelFrame(right_panel_top, text="Timeline Sequence")
        timeline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # 时间线列表框 + 滚动条
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

        # ---------------- 相机管理 ----------------
        camera_mgmt_frame = ttk.LabelFrame(right_panel_top, text="Camera Management")
        camera_mgmt_frame.grid(row=0, column=1, sticky="nsew")

        # 相机列表框 + 滚动条
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

        # 相机预览区域
        self.video_container = ttk.LabelFrame(right_panel, text="Camera Previews")
        self.video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(1, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(1, weight=1)

        self.speed_encoder_gui = SpeedEncoderGUI(self.root, right_panel)

        # 状态栏
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
        
        # 显示/隐藏速度图表
        if self.enable_speed_sensor.get():
            if self.speed_encoder_gui:
                self.speed_encoder_gui.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            if self.speed_encoder_gui:
                self.speed_encoder_gui.plot_frame.pack_forget()

    def on_type_change(self, event=None):
        # 根据选择的区块类型调整可见的输入字段
        btype = self.block_type.get()
        if btype == "Sound+Shock":
            # 隐藏单一持续时间字段
            self.duration_frame.grid_remove()
            # 显示Sound+Shock字段
            self.shock_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        else:
            # 隐藏Sound+Shock特定字段
            self.shock_frame.grid_remove()
            # 显示单一持续时间字段
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
            # 更新选中类型的字段可见性
            self.on_type_change()
            if btype == "Sound+Shock":
                # 条目格式: (类型, sound_dur, shock_lead)
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
        
        # 为不同的相机类型创建框架
        hk_frame = ttk.LabelFrame(popup, text="HIK Camera Parameters")
        hk_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        opencv_frame = ttk.LabelFrame(popup, text="OpenCV Camera Parameters")
        opencv_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        common_frame = ttk.Frame(popup)
        common_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # HIK相机特定参数
        ttk.Label(hk_frame, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        exposure_var = tk.DoubleVar(value=self.exposure_time.get())
        ttk.Entry(hk_frame, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(hk_frame, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        sample_rate_var = tk.DoubleVar(value=self.sample_rate.get())
        ttk.Entry(hk_frame, textvariable=sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # OpenCV相机特定参数
        camera_list = list_available_cameras()
        ttk.Label(opencv_frame, text="Camera Device:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        camera_var = tk.StringVar()
        camera_combo = ttk.Combobox(opencv_frame, textvariable=camera_var, values=camera_list, width=30)
        camera_combo.grid(row=0, column=1, padx=5, pady=5)
        if camera_list:
            camera_combo.current(0)

        # 两种相机类型的通用参数
        ttk.Label(common_frame, text="Resolution:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        resolution_var = tk.StringVar(value=self.resolution_choice.get())
        resolution_menu = ttk.OptionMenu(common_frame, resolution_var, resolution_var.get(), 
                                       "1920x1080","1280x1024", "1280x720", "640x480")
        resolution_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(common_frame, text="Recording FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        record_rate_var = tk.DoubleVar(value=self.record_rate.get())
        ttk.Entry(common_frame, textvariable=record_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # 切换参数可见性的函数
        def toggle_parameters():
            if camera_type.get() == "HKCamera":
                hk_frame.grid()
                opencv_frame.grid_remove()
            else:
                hk_frame.grid_remove()
                opencv_frame.grid()
        
        camera_type_combo.bind("<<ComboboxSelected>>", lambda e: toggle_parameters())
        toggle_parameters()  # 初始化可见性

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
        """移除选中的相机"""
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
        """编辑选中相机的参数"""
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
        """为相机创建预览"""
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
        """更新相机预览图像"""
        if not self.preview_active:  # 新增：检查预览是否激活
            return
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
        """启动所有相机预览"""
        self.preview_active = True  # 新增
        for camera in self.cameras:
            try:
                camera.start_preview(self.update_image)
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to start preview: {str(e)}")
        self.recording_label.config(text="Status: Previewing All Cameras")

    def stop_preview_all(self):
        """停止所有相机预览"""
        self.preview_active = False  # 新增
        for camera in self.cameras:
            camera.stop_preview()
        self.recording_label.config(text="Status: Preview Stopped")

    def update_timer(self):
        if any(camera.recording for camera in self.cameras):
            elapsed = time.time() - self.video_start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            self.time_label.config(text=f"{hours:02d}:{mins:02d}:{secs:02d}")
            timer_id = self.root.after(1000, self.update_timer)
            self.active_after_ids.append(timer_id)  # 跟踪定时器ID

    def cancel_all_after_tasks(self):
        """取消所有挂起的after任务"""
        for after_id in self.active_after_ids:
            self.root.after_cancel(after_id)
        self.active_after_ids = []

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
        if self.enable_speed_sensor.get() and self.speed_encoder_gui:
            try:
                self.speed_encoder_gui.input_rate = self.speed_input_rate.get()
                
                speed_file = os.path.join(self.save_dir, f"{prefix}_speed_{timestamp}.bin")
                if self.speed_encoder_gui.start_tracking(speed_file):
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

        if speed_sensor_enabled and self.speed_encoder_gui:
            self.speed_encoder_gui.stop_tracking()
            end_time = time.time()
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
        self.cancel_all_after_tasks()
        for camera in self.cameras:
            camera.stop_preview()
            camera.stop_record()
            camera.close()

        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")
            
        if self.speed_encoder_gui and self.speed_encoder_gui.running:
            self.speed_encoder_gui.active = False
            self.speed_encoder_gui.stop_tracking()

        self.root.destroy()
    
    def exit(self):
        self.cancel_all_after_tasks()
        for camera in self.cameras:
            camera.stop_preview()
            camera.close()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")
            
        if self.speed_encoder_gui and self.speed_encoder_gui.running:
            self.speed_encoder_gui.stop_tracking()
            self.speed_encoder_gui.active = False
            
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FearTrainingGUI(root)
    root.mainloop()