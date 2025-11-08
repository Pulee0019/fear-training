import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import os

class IRCameraControl:
    def __init__(self, root):
        self.root = root
        self.root.title("红外相机参数调节与录制")
        self.root.geometry("1200x700")
        
        # 相机变量
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = 0
        self.recording_duration = 0
        self.update_id = None
        
        # 创建界面
        self.create_widgets()
        
        # 初始化相机
        self.init_camera()
        
        # 开始更新预览
        self.update_preview()
        
        # 关闭窗口时的清理工作
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左侧参数控制框架
        control_frame = ttk.LabelFrame(main_frame, text="相机参数控制", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        # 右侧预览和录制框架
        preview_frame = ttk.LabelFrame(main_frame, text="相机预览", padding="10")
        preview_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # 参数控制部件
        params = [
            ("曝光", "CAP_PROP_EXPOSURE", -10, 10, 0),
            ("亮度", "CAP_PROP_BRIGHTNESS", 0, 100, 50),
            ("对比度", "CAP_PROP_CONTRAST", 0, 100, 50),
            ("饱和度", "CAP_PROP_SATURATION", 0, 100, 50),
            ("色调", "CAP_PROP_HUE", 0, 100, 50),
            ("增益", "CAP_PROP_GAIN", 0, 100, 0),
            ("锐度", "CAP_PROP_SHARPNESS", 0, 100, 50),
            ("伽马", "CAP_PROP_GAMMA", 100, 300, 100),
            ("白平衡", "CAP_PROP_WHITE_BALANCE_BLUE_U", 1000, 10000, 6500),
        ]
        
        self.param_vars = {}
        self.param_scale_widgets = {}
        
        for i, (name, prop, min_val, max_val, default) in enumerate(params):
            # 标签
            ttk.Label(control_frame, text=name).grid(row=i, column=0, sticky=tk.W, pady=5)
            
            # 滑动条
            var = tk.DoubleVar(value=default)
            scale = ttk.Scale(control_frame, from_=min_val, to=max_val, variable=var, 
                             orient=tk.HORIZONTAL, length=200,
                             command=lambda val, p=prop: self.on_param_change(p, val))
            scale.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=5)
            
            # 当前值显示
            value_label = ttk.Label(control_frame, text=str(default))
            value_label.grid(row=i, column=2, padx=(5, 0))
            
            # 保存引用
            self.param_vars[prop] = var
            self.param_scale_widgets[prop] = (scale, value_label)
        
        # 自动曝光控制
        ttk.Label(control_frame, text="自动曝光").grid(row=len(params), column=0, sticky=tk.W, pady=5)
        self.auto_exposure_var = tk.BooleanVar(value=False)
        auto_exposure_cb = ttk.Checkbutton(control_frame, variable=self.auto_exposure_var,
                                          command=self.toggle_auto_exposure)
        auto_exposure_cb.grid(row=len(params), column=1, sticky=tk.W, pady=5)
        
        # 分辨率设置
        ttk.Label(control_frame, text="分辨率").grid(row=len(params)+1, column=0, sticky=tk.W, pady=5)
        self.resolution_var = tk.StringVar(value="1920x1080")
        resolution_combo = ttk.Combobox(control_frame, textvariable=self.resolution_var,
                                       values=["640x480", "1280x720", "1920x1080"],
                                       state="readonly", width=15)
        resolution_combo.grid(row=len(params)+1, column=1, sticky=tk.W, pady=5)
        resolution_combo.bind("<<ComboboxSelected>>", self.change_resolution)
        
        # 帧率设置
        ttk.Label(control_frame, text="帧率").grid(row=len(params)+2, column=0, sticky=tk.W, pady=5)
        self.fps_var = tk.StringVar(value="30")
        fps_combo = ttk.Combobox(control_frame, textvariable=self.fps_var,
                                values=["15", "25", "30", "60"],
                                state="readonly", width=15)
        fps_combo.grid(row=len(params)+2, column=1, sticky=tk.W, pady=5)
        fps_combo.bind("<<ComboboxSelected>>", self.change_fps)
        
        # 保存/加载设置按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=len(params)+3, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="保存设置", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="加载设置", command=self.load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="默认设置", command=self.default_settings).pack(side=tk.LEFT, padx=5)
        
        # 预览区域
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 录制控制区域
        record_frame = ttk.Frame(preview_frame)
        record_frame.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
        self.record_button = ttk.Button(record_frame, text="开始录制", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        self.recording_time_var = tk.StringVar(value="00:00:00")
        self.recording_time_label = ttk.Label(record_frame, textvariable=self.recording_time_var)
        self.recording_time_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(record_frame, text="截图", command=self.take_screenshot).pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def init_camera(self):
        try:
            # 尝试打开相机
            self.cap = cv2.VideoCapture(1)
            
            if not self.cap.isOpened():
                # 如果索引1失败，尝试索引0
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                raise Exception("无法打开相机")
                
            # 设置初始分辨率和帧率
            self.change_resolution()
            self.change_fps()
            
            # 设置手动曝光模式
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            self.status_var.set("相机已连接")
            
        except Exception as e:
            messagebox.showerror("错误", f"初始化相机失败: {str(e)}")
            self.status_var.set("相机初始化失败")
    
    def on_param_change(self, prop, value):
        if self.cap is None or not self.cap.isOpened():
            return
            
        # 更新值显示
        scale, label = self.param_scale_widgets[prop]
        label.config(text=f"{float(value):.2f}")
        
        # 设置相机参数
        try:
            prop_id = getattr(cv2, prop)
            self.cap.set(prop_id, float(value))
        except Exception as e:
            print(f"设置参数 {prop} 失败: {str(e)}")
    
    def toggle_auto_exposure(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        if self.auto_exposure_var.get():
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动曝光
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光
    
    def change_resolution(self, event=None):
        if self.cap is None or not self.cap.isOpened():
            return
            
        width, height = map(int, self.resolution_var.get().split('x'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def change_fps(self, event=None):
        if self.cap is None or not self.cap.isOpened():
            return
            
        fps = int(self.fps_var.get())
        self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def update_preview(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 调整预览大小以适应窗口
                height, width = frame.shape[:2]
                max_size = 600
                if width > max_size:
                    scale = max_size / width
                    new_width = max_size
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # 转换为Tkinter可用的格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # 更新预览
                self.preview_label.configure(image=imgtk)
                self.preview_label.image = imgtk
                
                # 如果正在录制，写入视频帧
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # 更新录制时间
                    self.recording_duration = time.time() - self.recording_start_time
                    hours = int(self.recording_duration // 3600)
                    minutes = int((self.recording_duration % 3600) // 60)
                    seconds = int(self.recording_duration % 60)
                    self.recording_time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # 继续更新预览
        self.update_id = self.root.after(30, self.update_preview)
    
    def toggle_recording(self):
        if self.is_recording:
            # 停止录制
            self.is_recording = False
            self.record_button.config(text="开始录制")
            
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                
            self.status_var.set(f"录制已停止，视频已保存")
            
        else:
            # 开始录制
            file_path = filedialog.asksaveasfilename(
                defaultextension=".avi",
                filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*")]
            )
            
            if file_path:
                # 获取当前分辨率
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                
                # 创建视频编写器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                
                self.is_recording = True
                self.recording_start_time = time.time()
                self.recording_duration = 0
                self.record_button.config(text="停止录制")
                self.status_var.set("正在录制...")
    
    def take_screenshot(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".jpg",
                    filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
                )
                
                if file_path:
                    cv2.imwrite(file_path, frame)
                    self.status_var.set(f"截图已保存: {os.path.basename(file_path)}")
    
    def save_settings(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            settings = {}
            for prop, var in self.param_vars.items():
                settings[prop] = var.get()
            
            settings["resolution"] = self.resolution_var.get()
            settings["fps"] = self.fps_var.get()
            settings["auto_exposure"] = self.auto_exposure_var.get()
            
            import json
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.status_var.set(f"设置已保存: {os.path.basename(file_path)}")
    
    def load_settings(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                for prop, value in settings.items():
                    if prop in self.param_vars:
                        self.param_vars[prop].set(value)
                        self.on_param_change(prop, value)
                    elif prop == "resolution":
                        self.resolution_var.set(value)
                        self.change_resolution()
                    elif prop == "fps":
                        self.fps_var.set(value)
                        self.change_fps()
                    elif prop == "auto_exposure":
                        self.auto_exposure_var.set(value)
                        self.toggle_auto_exposure()
                
                self.status_var.set(f"设置已加载: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载设置失败: {str(e)}")
    
    def default_settings(self):
        # 重置所有参数到默认值
        self.resolution_var.set("1920x1080")
        self.change_resolution()
        
        self.fps_var.set("30")
        self.change_fps()
        
        self.auto_exposure_var.set(False)
        self.toggle_auto_exposure()
        
        # 重置所有参数滑块
        default_values = {
            "CAP_PROP_EXPOSURE": 0,
            "CAP_PROP_BRIGHTNESS": 50,
            "CAP_PROP_CONTRAST": 50,
            "CAP_PROP_SATURATION": 50,
            "CAP_PROP_HUE": 50,
            "CAP_PROP_GAIN": 0,
            "CAP_PROP_SHARPNESS": 50,
            "CAP_PROP_GAMMA": 100,
            "CAP_PROP_WHITE_BALANCE_BLUE_U": 6500,
        }
        
        for prop, value in default_values.items():
            if prop in self.param_vars:
                self.param_vars[prop].set(value)
                self.on_param_change(prop, value)
        
        self.status_var.set("已恢复默认设置")
    
    def on_closing(self):
        # 停止预览更新
        if self.update_id:
            self.root.after_cancel(self.update_id)
        
        # 停止录制
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
        
        # 释放相机
        if self.cap is not None:
            self.cap.release()
        
        # 关闭窗口
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = IRCameraControl(root)
    root.mainloop()