import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
import queue
import os
import PySpin

# 尝试导入HIK相机SDK
try:
    from hikvisionapi import Client
    import ctypes
    from ctypes import *
    # 假设HIK SDK的dll文件在系统路径中
    # 实际使用时可能需要指定路径
    # ctypes.CDLL("HCNetSDK.dll")
    HAS_HIK = True
except:
    HAS_HIK = False
    print("警告: HIK相机SDK导入失败，HIK相机功能将不可用")

class FLIRCamera:
    """FLIR相机类，用于控制和录制FLIR相机"""
    
    def __init__(self, camera_id=0, width=640, height=480, exposure_time=10000):
        self.camera_id = camera_id
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = None
        self.running = False
        self.previewing = False
        self.recording = False
        self.out = None
        self.current_fps = 0
        self._frame_count = 0
        self._fps_last_time = time.time()
        self.display_queue = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=100)
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.sample_rate = 30.0  # 采样率
        self.record_rate = 30.0  # 录制率
        self.device_serial = None
        
        if len(self.cam_list) > camera_id:
            self.cam = self.cam_list[camera_id]
            self.cam.Init()
            self.device_serial = self.cam.TLDevice.DeviceSerialNumber.GetValue()
            self.set_parameters(width, height, exposure_time, self.sample_rate, self.record_rate)
        else:
            raise ValueError(f"无法找到ID为{camera_id}的FLIR相机")
    
    def set_parameters(self, width, height, exposure_time, sample_rate, record_rate):
        """设置相机参数"""
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.sample_rate = sample_rate
        self.record_rate = record_rate
        
        if self.cam:
            # 设置宽度和高度
            width_node = self.cam.Width
            if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
                max_width = width_node.GetMax()
                self.width = min(width, max_width)
                width_node.SetValue(self.width)
            
            height_node = self.cam.Height
            if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
                max_height = height_node.GetMax()
                self.height = min(height, max_height)
                height_node.SetValue(self.height)
            
            # 设置曝光时间
            exposure_node = self.cam.ExposureTime
            if PySpin.IsAvailable(exposure_node) and PySpin.IsWritable(exposure_node):
                min_exposure = exposure_node.GetMin()
                max_exposure = exposure_node.GetMax()
                self.exposure_time = min(max(exposure_time, min_exposure), max_exposure)
                exposure_node.SetValue(self.exposure_time)
            
            # 设置帧率
            fps_node = self.cam.AcquisitionFrameRate
            if PySpin.IsAvailable(fps_node) and PySpin.IsWritable(fps_node):
                min_fps = fps_node.GetMin()
                max_fps = fps_node.GetMax()
                self.sample_rate = min(max(sample_rate, min_fps), max_fps)
                fps_node.SetValue(self.sample_rate)
            
            fps_enable = self.cam.AcquisitionFrameRateEnable
            if PySpin.IsAvailable(fps_enable) and PySpin.IsWritable(fps_enable):
                fps_enable.SetValue(True)
    
    def open(self):
        """打开相机"""
        if self.cam and not self.running:
            self.cam.BeginAcquisition()
            self.running = True
            self.preview_thread = threading.Thread(target=self._acquisition_loop)
            self.preview_thread.daemon = True
            self.preview_thread.start()
    
    def close(self):
        """关闭相机"""
        self.previewing = False
        self.recording = False
        self.running = False
        
        if self.out:
            self.out.release()
            self.out = None
            
        if self.cam:
            try:
                self.cam.EndAcquisition()
            except:
                pass
            
            try:
                self.cam.DeInit()
            except:
                pass
            
            del self.cam
            
        self.cam_list.Clear()
        self.system.ReleaseInstance()
    
    def start_preview(self):
        """开始预览"""
        self.previewing = True
    
    def stop_preview(self):
        """停止预览"""
        self.previewing = False
    
    def start_recording(self, output_file):
        """开始录制"""
        if not self.recording and self.running:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_file, fourcc, self.record_rate, (self.width, self.height))
            self.recording = True
            self.record_thread = threading.Thread(target=self.record_loop)
            self.record_thread.daemon = True
            self.record_thread.start()
    
    def stop_recording(self):
        """停止录制"""
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
    
    def _acquisition_loop(self):
        """相机采集循环"""
        try:
            while self.running:
                if self.previewing or self.recording:
                    image_result = self.cam.GetNextImage()
                    if image_result.IsIncomplete():
                        continue
                    
                    img = image_result.GetNDArray()
                    image_result.Release()
                    
                    # 转换为BGR格式
                    if len(img.shape) == 2:  # 黑白图像
                        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:  # 彩色图像
                        frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                    
                    # 更新FPS计算
                    self._frame_count += 1
                    current_time = time.time()
                    if current_time - self._fps_last_time >= 1.0:
                        self.current_fps = self._frame_count / (current_time - self._fps_last_time)
                        self._frame_count = 0
                        self._fps_last_time = current_time
                    
                    # 放入预览队列
                    if self.previewing and not self.display_queue.full():
                        self.display_queue.put(frame)
                    
                    # 放入录制队列
                    if self.recording and not self.frame_queue.full():
                        self.frame_queue.put((time.time(), frame.copy()))
                
                else:
                    time.sleep(0.01)  # 减少CPU使用率
        
        except Exception as e:
            print(f"FLIR相机ID {self.camera_id} 采集错误:", str(e))
        finally:
            self.running = False
            self.previewing = False
            self.recording = False
    
    def record_loop(self):
        """录制循环"""
        interval = 1.0 / self.record_rate
        next_time = time.time()
        
        while self.recording:
            try:
                if not self.frame_queue.empty():
                    timestamp, frame = self.frame_queue.get()
                    self.out.write(frame)
                
                # 控制录制帧率
                next_time += interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            except Exception as e:
                print(f"FLIR相机ID {self.camera_id} 录制错误:", str(e))

class HIKCamera:
    """HIK相机类，用于控制和录制HIK相机"""
    
    def __init__(self, camera_id=0, ip="192.168.1.64", username="admin", password="password", 
                 width=640, height=480, fps=30):
        if not HAS_HIK:
            raise RuntimeError("HIK相机SDK未正确导入，无法使用HIK相机功能")
        
        self.camera_id = camera_id
        self.ip = ip
        self.username = username
        self.password = password
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.previewing = False
        self.recording = False
        self.out = None
        self.current_fps = 0
        self._frame_count = 0
        self._fps_last_time = time.time()
        self.display_queue = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=100)
        self.device_info = None
        
        # 初始化HIK相机
        self.init_camera()
    
    def init_camera(self):
        """初始化HIK相机连接"""
        try:
            self.client = Client(f"http://{self.ip}", self.username, self.password, timeout=30)
            # 获取设备信息
            self.device_info = self.client.System.deviceInfo()
            # 尝试打开预览流
            self.cap = cv2.VideoCapture(f"rtsp://{self.username}:{self.password}@{self.ip}:554/Streaming/Channels/1")
            if not self.cap.isOpened():
                raise ValueError(f"无法连接到HIK相机: {self.ip}")
            
            # 设置分辨率
            if self.width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 获取实际分辨率和帧率
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
        except Exception as e:
            print(f"HIK相机初始化错误 ({self.ip}):", str(e))
            raise
    
    def open(self):
        """打开相机"""
        if not self.running:
            self.running = True
            self.preview_thread = threading.Thread(target=self._capture_loop)
            self.preview_thread.daemon = True
            self.preview_thread.start()
    
    def close(self):
        """关闭相机"""
        self.previewing = False
        self.recording = False
        self.running = False
        
        if self.out:
            self.out.release()
            self.out = None
            
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
    
    def start_preview(self):
        """开始预览"""
        self.previewing = True
    
    def stop_preview(self):
        """停止预览"""
        self.previewing = False
    
    def start_recording(self, output_file):
        """开始录制"""
        if not self.recording and self.running:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
            self.recording = True
    
    def stop_recording(self):
        """停止录制"""
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
    
    def _capture_loop(self):
        """相机捕获循环"""
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)  # 读取失败时等待一段时间
                    continue
                
                # 更新FPS计算
                self._frame_count += 1
                current_time = time.time()
                if current_time - self._fps_last_time >= 1.0:
                    self.current_fps = self._frame_count / (current_time - self._fps_last_time)
                    self._frame_count = 0
                    self._fps_last_time = current_time
                
                # 放入预览队列
                if self.previewing and not self.display_queue.full():
                    self.display_queue.put(frame)
                
                # 放入录制队列
                if self.recording and not self.frame_queue.full():
                    self.frame_queue.put((time.time(), frame.copy()))
                
                # 如果没有预览或录制，适当降低CPU使用率
                if not self.previewing and not self.recording:
                    time.sleep(0.05)
        
        except Exception as e:
            print(f"HIK相机 ({self.ip}) 捕获错误:", str(e))
        finally:
            self.running = False
            self.previewing = False
            self.recording = False

class OpenCVCamera:
    """OpenCV相机类，用于控制和录制普通USB相机"""
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.previewing = False
        self.recording = False
        self.out = None
        self.current_fps = 0
        self._frame_count = 0
        self._fps_last_time = time.time()
        self.display_queue = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=100)
        
        # 尝试打开相机
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开ID为{camera_id}的OpenCV相机")
        
        # 设置分辨率和帧率
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 获取实际分辨率和帧率
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def open(self):
        """打开相机"""
        if not self.running:
            self.running = True
            self.preview_thread = threading.Thread(target=self._capture_loop)
            self.preview_thread.daemon = True
            self.preview_thread.start()
    
    def close(self):
        """关闭相机"""
        self.previewing = False
        self.recording = False
        self.running = False
        
        if self.out:
            self.out.release()
            self.out = None
            
        if self.cap:
            self.cap.release()
    
    def start_preview(self):
        """开始预览"""
        self.previewing = True
    
    def stop_preview(self):
        """停止预览"""
        self.previewing = False
    
    def start_recording(self, output_file):
        """开始录制"""
        if not self.recording and self.running:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
            self.recording = True
    
    def stop_recording(self):
        """停止录制"""
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
    
    def _capture_loop(self):
        """相机捕获循环"""
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)  # 读取失败时等待一段时间
                    continue
                
                # 更新FPS计算
                self._frame_count += 1
                current_time = time.time()
                if current_time - self._fps_last_time >= 1.0:
                    self.current_fps = self._frame_count / (current_time - self._fps_last_time)
                    self._frame_count = 0
                    self._fps_last_time = current_time
                
                # 放入预览队列
                if self.previewing and not self.display_queue.full():
                    self.display_queue.put(frame)
                
                # 放入录制队列
                if self.recording and not self.frame_queue.full():
                    self.frame_queue.put((time.time(), frame.copy()))
                
                # 如果没有预览或录制，适当降低CPU使用率
                if not self.previewing and not self.recording:
                    time.sleep(0.05)
        
        except Exception as e:
            print(f"OpenCV相机ID {self.camera_id} 捕获错误:", str(e))
        finally:
            self.running = False
            self.previewing = False
            self.recording = False

def list_available_cameras():
    """列出所有可用的相机，包括FLIR、HIK和OpenCV相机"""
    camera_list = []
    
    # 检测FLIR相机
    try:
        system = PySpin.System.GetInstance()
        flir_cam_list = system.GetCameras()
        flir_count = flir_cam_list.GetSize()
        
        for i in range(flir_count):
            cam = flir_cam_list[i]
            cam.Init()
            try:
                serial = cam.TLDevice.DeviceSerialNumber.GetValue()
                model = cam.TLDevice.DeviceModelName.GetValue()
                camera_list.append(f"FLIR: {model} (SN: {serial}, ID: {i})")
            except:
                camera_list.append(f"FLIR: 相机 {i}")
            cam.DeInit()
            del cam
        
        flir_cam_list.Clear()
        system.ReleaseInstance()
    except Exception as e:
        print("检测FLIR相机时出错:", str(e))
    
    # 检测HIK相机
    if HAS_HIK:
        try:
            # 这里应该实现HIK相机的自动发现
            # 简化版：假设用户已经配置了HIK相机
            # 实际应用中可能需要实现网络扫描功能
            hik_cameras = [
                {"id": 0, "ip": "192.168.1.64", "model": "DS-2CD2385FWD-I"}
            ]
            
            for cam_info in hik_cameras:
                camera_list.append(f"HIK: {cam_info['model']} ({cam_info['ip']}, ID: {cam_info['id']})")
        except Exception as e:
            print("检测HIK相机时出错:", str(e))
    
    # 检测OpenCV相机
    try:
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                camera_list.append(f"OpenCV: 相机 {index}")
            cap.release()
            index += 1
    except:
        pass
    
    return camera_list

class CameraApp:
    """相机应用主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("多相机录制系统")
        self.root.geometry("1000x600")
        
        self.cameras = {}  # 存储已打开的相机
        self.current_camera = None
        
        # 创建UI
        self.create_widgets()
        
        # 初始化相机列表
        self.refresh_camera_list()
    
    def create_widgets(self):
        """创建UI组件"""
        # 顶部框架 - 相机选择
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="选择相机:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(top_frame, textvariable=self.camera_var, values=[], width=50)
        self.camera_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(top_frame, text="刷新相机列表", command=self.refresh_camera_list).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(top_frame, text="打开相机", command=self.open_selected_camera).grid(row=0, column=3, padx=5, pady=5)
        
        # 主内容框架
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧 - 预览窗口
        preview_frame = ttk.LabelFrame(content_frame, text="预览", padding=10)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧 - 控制面板
        control_frame = ttk.LabelFrame(content_frame, text="控制", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 参数设置
        param_frame = ttk.LabelFrame(control_frame, text="相机参数", padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text="宽度:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.width_var = tk.StringVar(value="640")
        ttk.Entry(param_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="高度:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.height_var = tk.StringVar(value="480")
        ttk.Entry(param_frame, textvariable=self.height_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="帧率:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.fps_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(control_frame, padding=10)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.preview_button = ttk.Button(button_frame, text="开始预览", command=self.toggle_preview)
        self.preview_button.pack(fill=tk.X, pady=5)
        
        self.record_button = ttk.Button(button_frame, text="开始录制", command=self.toggle_recording)
        self.record_button.pack(fill=tk.X, pady=5)
        
        ttk.Label(button_frame, text="录制文件名:").pack(anchor=tk.W, pady=5)
        self.filename_var = tk.StringVar(value="output.avi")
        ttk.Entry(button_frame, textvariable=self.filename_var).pack(fill=tk.X, pady=5)
        
        # 状态信息
        status_frame = ttk.LabelFrame(control_frame, text="状态", padding=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="未连接相机")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.fps_var).pack(anchor=tk.W)
    
    def refresh_camera_list(self):
        """刷新相机列表"""
        camera_list = list_available_cameras()
        self.camera_combo['values'] = camera_list
        if camera_list:
            self.camera_combo.current(0)
    
    def open_selected_camera(self):
        """打开选中的相机"""
        camera_info = self.camera_var.get()
        if not camera_info:
            messagebox.showwarning("警告", "请先选择一个相机")
            return
        
        # 关闭当前相机（如果有）
        if self.current_camera and self.current_camera in self.cameras:
            self.cameras[self.current_camera].close()
            del self.cameras[self.current_camera]
        
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = int(self.fps_var.get())
            
            # 解析相机信息
            if camera_info.startswith("FLIR:"):
                # FLIR相机
                parts = camera_info.split(", ID: ")
                camera_id = int(parts[-1].strip(")"))
                self.cameras[camera_info] = FLIRCamera(camera_id, width, height)
            elif camera_info.startswith("HIK:"):
                # HIK相机
                if not HAS_HIK:
                    messagebox.showerror("错误", "HIK相机SDK未正确导入，无法使用HIK相机功能")
                    return
                
                parts = camera_info.split("(")
                ip = parts[1].split(",")[0]
                camera_id = int(parts[1].split("ID: ")[1].strip(")"))
                self.cameras[camera_info] = HIKCamera(camera_id, ip, "admin", "password", width, height, fps)
            elif camera_info.startswith("OpenCV:"):
                # OpenCV相机
                camera_id = int(camera_info.split("相机 ")[1])
                self.cameras[camera_info] = OpenCVCamera(camera_id, width, height, fps)
            else:
                messagebox.showerror("错误", "不支持的相机类型")
                return
            
            self.current_camera = camera_info
            self.cameras[camera_info].open()
            self.status_var.set(f"已连接: {camera_info}")
            self.preview_button.config(state=tk.NORMAL)
            self.record_button.config(state=tk.NORMAL)
            
            # 开始更新预览
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("错误", f"打开相机失败: {str(e)}")
            if camera_info in self.cameras:
                self.cameras[camera_info].close()
                del self.cameras[camera_info]
    
    def toggle_preview(self):
        """切换预览状态"""
        if not self.current_camera or self.current_camera not in self.cameras:
            return
        
        camera = self.cameras[self.current_camera]
        
        if camera.previewing:
            camera.stop_preview()
            self.preview_button.config(text="开始预览")
        else:
            camera.start_preview()
            self.preview_button.config(text="停止预览")
    
    def toggle_recording(self):
        """切换录制状态"""
        if not self.current_camera or self.current_camera not in self.cameras:
            return
        
        camera = self.cameras[self.current_camera]
        filename = self.filename_var.get()
        
        if camera.recording:
            camera.stop_recording()
            self.record_button.config(text="开始录制")
            self.status_var.set(f"已停止录制: {filename}")
        else:
            if not filename:
                messagebox.showwarning("警告", "请输入录制文件名")
                return
            
            # 确保文件路径存在
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            camera.start_recording(filename)
            self.record_button.config(text="停止录制")
            self.status_var.set(f"正在录制: {filename}")
    
    def update_preview(self):
        """更新预览窗口"""
        if self.current_camera and self.current_camera in self.cameras:
            camera = self.cameras[self.current_camera]
            
            if camera.previewing and not camera.display_queue.empty():
                try:
                    frame = camera.display_queue.get_nowait()
                    
                    # 调整图像大小以适应预览窗口
                    preview_width = self.preview_label.winfo_width()
                    preview_height = self.preview_label.winfo_height()
                    
                    if preview_width > 10 and preview_height > 10:
                        frame = cv2.resize(frame, (preview_width, preview_height))
                    
                    # 转换为Tkinter可用的格式
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(image=img)
                    
                    # 更新预览标签
                    self.preview_label.config(image=img)
                    self.preview_label.image = img
                
                except Exception as e:
                    print("更新预览时出错:", str(e))
            
            # 更新FPS显示
            self.fps_var.set(f"FPS: {camera.current_fps:.1f}")
        
        # 继续更新预览
        self.root.after(30, self.update_preview)

if __name__ == "__main__":
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        import cv2
        import numpy as np
        import threading
        import time
        import queue
        import os
        import PySpin
        
        # 尝试导入HIK相机SDK
        try:
            from hikvisionapi import Client
            import ctypes
            HAS_HIK = True
        except:
            HAS_HIK = False
            print("警告: HIK相机SDK导入失败，HIK相机功能将不可用")
        
        root = tk.Tk()
        app = CameraApp(root)
        root.mainloop()
    
    except Exception as e:
        print("程序启动错误:", str(e))    