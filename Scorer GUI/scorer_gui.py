import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from collections import defaultdict
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import maxvit_t, MaxVit_T_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from itertools import product
import kornia
import kornia.augmentation as K
from tqdm import tqdm
import warnings
import threading
import queue
# 忽略PIL的EXIF警告
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FontManager:
    """统一管理字体和padding的类，区分不同UI元素的字体和间距"""
    def __init__(self):
        # 定义不同类型的字体
        self.fonts = {
            "title": ("Arial", 24, "bold"),
            "label": ("Arial", 20, "normal"),
            "button": ("Arial", 20, "normal"),
            "entry": ("Arial", 20, "normal"),
            "listbox": ("Arial", 20, "normal"),
            "chart_label": ("Arial", 20, "normal"),
        }
        # 定义不同类型的padding
        self.paddings = {
            "main": 20,
            "frame": 10,
            "widget": 4,
        }
        # 字体和padding缩放因子
        self.scale_factor = 1.0

    def set_scale(self, scale):
        """设置字体和padding缩放比例"""
        self.scale_factor = scale
        # 重新计算所有字体大小
        scaled_fonts = {}
        for name, (family, size, weight) in self.fonts.items():
            scaled_size = int(size * self.scale_factor)
            scaled_fonts[name] = (family, scaled_size, weight)
        self.fonts = scaled_fonts
        
        # 重新计算所有padding大小
        scaled_paddings = {}
        for name, value in self.paddings.items():
            scaled_value = int(value * self.scale_factor)
            scaled_paddings[name] = scaled_value
        self.paddings = scaled_paddings

    def get_font(self, element_type="label"):
        """根据元素类型获取字体"""
        # 如果指定的类型不存在，返回默认的label字体
        return self.fonts.get(element_type, self.fonts["label"])
        
    def get_padding(self, element_type="main"):
        """根据元素类型获取padding"""
        # 如果指定的类型不存在，返回默认的main padding
        return self.paddings.get(element_type, self.paddings["main"])

    def update_font(self, element_type, family=None, size=None, weight=None):
        """动态更新特定类型的字体"""
        if element_type in self.fonts:
            current_family, current_size, current_weight = self.fonts[element_type]
            new_family = family if family is not None else current_family
            new_size = size if size is not None else current_size
            new_weight = weight if weight is not None else current_weight
            self.fonts[element_type] = (new_family, new_size, new_weight)
            
    def update_padding(self, element_type, value):
        """动态更新特定类型的padding"""
        if element_type in self.paddings:
            self.paddings[element_type] = value


class BaseComponent:
    """所有组件的基类"""
    def __init__(self, parent, font_manager):
        self.parent = parent
        self.font_manager = font_manager


class FolderSelector(BaseComponent):
    """文件夹选择组件"""
    def __init__(self, parent, font_manager):
        super().__init__(parent, font_manager)
        self.image_folder = ""
        self.json_file = ""
        self.folder_var = tk.StringVar()
        self.json_var = tk.StringVar()
        self.info_var = None # 初始化为 None，后续会被赋值

    def create_ui(self, parent_frame, info_var):
        self.info_var = info_var # 获取主UI的info_var引用
        
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        folder_frame = ttk.LabelFrame(parent_frame, text="文件夹设置", padding="5", style="Custom.TLabelframe")
        folder_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="图片目录:", font=self.font_manager.get_font("label")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        # ttk.Entry 不支持 font 参数，使用 style
        entry_style = ttk.Style()
        entry_style.configure("Custom.TEntry", font=self.font_manager.get_font("entry"))
        ttk.Entry(folder_frame, textvariable=self.folder_var, state='readonly', style="Custom.TEntry").grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(folder_frame, text="浏览", command=self.select_folder, style="Custom.TButton").grid(row=0, column=2)

        ttk.Label(folder_frame, text="JSON文件:", font=self.font_manager.get_font("label")).grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(folder_frame, textvariable=self.json_var, state='readonly', style="Custom.TEntry").grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(folder_frame, text="浏览", command=self.select_json, style="Custom.TButton").grid(row=1, column=2)

        # 配置按钮样式
        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=self.font_manager.get_font("button"))

        return folder_frame

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择图片目录")
        if folder:
            self.folder_var.set(folder)
            self.image_folder = folder
            # 自动查找或创建JSON文件
            self.auto_find_json()

    def select_json(self):
        json_file = filedialog.asksaveasfilename(
            title="选择JSON文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if json_file:
            self.json_var.set(json_file)
            self.json_file = json_file
            self.load_scores()

    def auto_find_json(self):
        """自动在图片目录中查找或创建JSON文件"""
        if not self.image_folder:
            return

        # 首先在图片目录中查找scores.json
        possible_json = os.path.join(self.image_folder, "scores.json")
        if os.path.exists(possible_json):
            self.json_file = possible_json
            self.json_var.set(possible_json)
            self.load_scores()
        else:
            # 如果没有找到，创建新的JSON文件
            self.json_file = possible_json
            self.json_var.set(possible_json)
            # self.scores = {} # 这个应该在ImageScorerUI中初始化
            # self.save_scores() # 这个应该在ImageScorerUI中调用
            # 现在只设置路径，加载会在load_scores中进行
            if self.info_var:
                self.info_var.set("未找到现有评分文件，将创建新的: scores.json")

    def load_scores(self):
        """加载评分数据"""
        if not self.json_file:
            if self.info_var:
                self.info_var.set("请先选择JSON文件")
            return
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    scores = json.load(f) # 临时加载，避免修改self.parent.scores
                self.parent.scores = scores # 将加载的数据赋给主UI
                if self.info_var:
                    self.info_var.set(f"已加载 {len(self.parent.scores)} 个评分")
            except Exception as e:
                messagebox.showerror("错误", f"加载JSON文件失败: {e}")
                self.parent.scores = {}
        else:
            self.parent.scores = {}
            self.save_scores()

    def save_scores(self):
        """保存评分数据"""
        if not self.json_file:
            if self.info_var:
                self.info_var.set("请先选择JSON文件")
            return
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.parent.scores, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("错误", f"保存JSON文件失败: {e}")


class ControlButtons(BaseComponent):
    """控制按钮组件"""
    def __init__(self, parent, font_manager, auto_next_var, zoom_var, filter_toggle_func):
        super().__init__(parent, font_manager)
        self.auto_next_var = auto_next_var
        self.zoom_var = zoom_var
        self.filter_toggle_func = filter_toggle_func

    def create_ui(self, parent_frame):
        control_frame = ttk.Frame(parent_frame)
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(control_frame, text="加载图片", command=self.parent.load_images, style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="上一张 (Q)", command=self.parent.previous_image, style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="下一张 (E)", command=self.parent.next_image, style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="保存并退出 (W)", command=self.parent.save_and_exit, style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="显示/隐藏过滤器 (F)", command=self.filter_toggle_func, style="Custom.TButton").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Button(control_frame, text="展开/收起预处理", command=self.parent.preprocessing_ui.toggle_expansion, style="Custom.TButton").pack(side=tk.LEFT, padx=(5, 0))
        # 选项区域
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(side=tk.LEFT, padx=(20, 0))

        ttk.Checkbutton(options_frame, text="打分后自动下一张", 
                       variable=self.auto_next_var, style="Custom.TCheckbutton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(options_frame, text="图片放大到窗口", 
                       variable=self.zoom_var, command=self.parent.display_current_image, style="Custom.TCheckbutton").pack(side=tk.LEFT)

        # 配置样式
        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=self.font_manager.get_font("button"))
        button_style.configure("Custom.TCheckbutton", font=self.font_manager.get_font("button")) # 复选框文字使用按钮字体

        return control_frame


class ImageViewer(BaseComponent):
    """图片查看器组件"""
    def __init__(self, parent, font_manager):
        super().__init__(parent, font_manager)
        self.canvas = None
        self.tk_img = None

    def create_ui(self, parent_frame):
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        image_frame = ttk.LabelFrame(parent_frame, text="图片显示", padding="5", style="Custom.TLabelframe")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # 创建画布用于显示图片
        self.canvas = tk.Canvas(image_frame, bg='white')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 添加滚动条
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        return image_frame

    def display_image(self, image_path, zoom_to_fit):
        """显示指定路径的图片"""
        try:
            # 使用OpenCV读取图片
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("无法读取图片")

            # 转换颜色空间 BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 获取画布尺寸
            canvas_width = self.canvas.winfo_width() - 20
            canvas_height = self.canvas.winfo_height() - 20

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600

            h, w = img.shape[:2]

            if zoom_to_fit:
                # 放大到窗口内显示，保持长宽比
                scale = min(canvas_width / w, canvas_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                # 原始大小或适当缩放
                scale = min(canvas_width / w, canvas_height / h, 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)

            # 调整图片大小
            img_resized = cv2.resize(img, (new_w, new_h))

            # 转换为PIL Image
            pil_img = Image.fromarray(img_resized)
            self.tk_img = ImageTk.PhotoImage(pil_img)

            # 清除画布并显示图片
            self.canvas.delete("all")
            self.canvas.create_image(10, 10, anchor=tk.NW, image=self.tk_img)

            # 更新滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        except Exception as e:
            messagebox.showerror("错误", f"显示图片失败: {e}")


class ScoreButtons(BaseComponent):
    """评分按钮组件"""
    def __init__(self, parent, font_manager, score_var, update_score_func):
        super().__init__(parent, font_manager)
        self.score_var = score_var
        self.update_score_func = update_score_func

    def create_ui(self, parent_frame, filter_toggle_func):
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        score_frame = ttk.LabelFrame(parent_frame, text="评分", padding="5", style="Custom.TLabelframe")
        score_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(score_frame, text="当前分数:", font=self.font_manager.get_font("label")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        # Entry 不支持 font，使用 style
        entry_style = ttk.Style()
        entry_style.configure("Custom.TEntry", font=self.font_manager.get_font("entry"))
        self.score_entry = ttk.Entry(score_frame, textvariable=self.score_var, width=10, font=self.font_manager.get_font("entry"))
        self.score_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        self.score_entry.bind('<Return>', self.update_score_func)

        # 快速评分按钮 (0-9，没有10)
        quick_score_frame = ttk.Frame(score_frame)
        quick_score_frame.grid(row=0, column=2, sticky=tk.W)

        for i in range(10):  # 0-9
            ttk.Button(quick_score_frame, text=str(i), 
                      command=lambda x=i: self.set_score(x), style="Custom.TButton").pack(side=tk.LEFT, padx=2)

        # 微调分数按钮
        adjust_frame = ttk.Frame(score_frame)
        adjust_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        ttk.Button(adjust_frame, text="+0.5 (↑)", 
                  command=lambda: self.adjust_score(0.5, auto_next=False), style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(adjust_frame, text="-0.5 (↓)", 
                  command=lambda: self.adjust_score(-0.5, auto_next=False), style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(adjust_frame, text="+0.1 (→)", 
                  command=lambda: self.adjust_score(0.1, auto_next=False), style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(adjust_frame, text="-0.1 (←)", 
                  command=lambda: self.adjust_score(-0.1, auto_next=False), style="Custom.TButton").pack(side=tk.LEFT)


        # 配置样式
        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=self.font_manager.get_font("button"), padding=1)
        # entry_style 已在上面配置

        return score_frame

    def set_score(self, score):
        """快速设置分数"""
        self.score_var.set(str(score))
        self.update_score_func()

    def adjust_score(self, delta, auto_next=True):
        """调整当前分数"""
        try:
            current_text = self.score_var.get()
            if current_text == "未评分":
                current_score = 0.0
            else:
                current_score = float(current_text)

            new_score = max(0.0, min(9.0, current_score + delta))  # 上限改为9.0
            self.score_var.set(f"{new_score:.1f}")

            # 立即更新分数到JSON
            image_path = self.parent.image_paths[self.parent.current_index]
            rel_path = os.path.relpath(image_path, self.parent.folder_selector.image_folder)
            self.parent.scores[rel_path] = new_score
            # 修复：调用 FolderSelector 的 save_scores 方法
            self.parent.folder_selector.save_scores()
            # 更新信息栏
            self.parent.update_info()

            # 更新过滤系统（重新计算分布和筛选）
            if self.parent.filter_system.filter_visible:
                self.parent.filter_system.update_score_distribution()
                self.parent.filter_system.update_filtered_images()

            # 只有明确要求auto_next时才自动下一张
            if auto_next and self.parent.auto_next_after_score.get():
                self.parent.next_image()
            else:
                self.parent.root.focus_set()

        except ValueError:
            pass


class InfoDisplay(BaseComponent):
    """信息显示组件"""
    def __init__(self, parent, font_manager):
        super().__init__(parent, font_manager)
        self.info_var = tk.StringVar(value="请选择图片目录开始评分")
        self.progress_var = tk.StringVar(value="0/0")

    def create_ui(self, parent_frame):
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        info_frame = ttk.LabelFrame(parent_frame, text="信息", padding="5", style="Custom.TLabelframe")
        info_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(info_frame, textvariable=self.info_var, font=self.font_manager.get_font("label")).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.progress_var, font=self.font_manager.get_font("label")).grid(row=0, column=1, sticky=tk.E)

        return info_frame


class ShortcutDisplay(BaseComponent):
    """快捷键显示组件"""
    def __init__(self, parent, font_manager):
        super().__init__(parent, font_manager)

    def create_ui(self, parent_frame):
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        shortcut_frame = ttk.LabelFrame(parent_frame, text="快捷键提示", padding="5", style="Custom.TLabelframe")
        shortcut_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))

        shortcuts = [
            "Q: 上一张图片 | E: 下一张图片 | W: 保存并退出 | F: 显示/隐藏过滤器",
            "0-9: 快速打分 | ↑/↓: ±0.5分 | ←/→: ±0.1分",
            "Enter: 确认分数 | ESC: 退出程序"
        ]

        for i, shortcut in enumerate(shortcuts):
            ttk.Label(shortcut_frame, text=shortcut, font=self.font_manager.get_font("label")).grid(row=i, column=0, sticky=tk.W)

        return shortcut_frame


class ScoreFilterSystem(BaseComponent):
    """评分过滤系统组件"""
    def __init__(self, parent, font_manager):
        super().__init__(parent, font_manager)
        self.root = parent.root
        self.filter_visible = False
        self.bin_count = 10
        self.min_filter_score = 0.0
        self.max_filter_score = 9.0
        self.filtered_images = []
        self.figure, self.ax = None, None
        self.canvas = None
        self.vline_left = None
        self.vline_right = None
        self.is_dragging_left = False
        self.is_dragging_right = False
        self.filter_frame = None

    def create_ui(self, parent_frame):
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        # 过滤器容器框架（初始隐藏）
        self.filter_frame = ttk.LabelFrame(parent_frame, text="评分过滤系统", padding="10", style="Custom.TLabelframe")
        # 放在右侧，跨越多行
        self.filter_frame.grid(row=2, column=3, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(0, 10))
        self.filter_frame.columnconfigure(0, weight=1)
        self.filter_frame.rowconfigure(1, weight=1) # 给图表区域分配更多权重
        self.filter_frame.rowconfigure(2, weight=1) # 给列表区域分配更多权重
        self.filter_frame.grid_remove()  # 初始隐藏

        # 区间数量调节
        bin_frame = ttk.Frame(self.filter_frame)
        bin_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Label(bin_frame, text="评分区间数:", font=self.font_manager.get_font("label")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.bin_slider = ttk.Scale(bin_frame, from_=2, to=30, variable=tk.IntVar(value=self.bin_count), 
                                   orient=tk.HORIZONTAL, command=self.on_bin_count_change)
        self.bin_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.bin_label = ttk.Label(bin_frame, text=str(self.bin_count), font=self.font_manager.get_font("label"))
        self.bin_label.grid(row=0, column=2, sticky=tk.W)

        # 评分分布图表
        self.figure, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.filter_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        self.canvas.figure = self.figure
        # 绑定图表拖拽事件
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 符合条件图片列表
        list_frame = ttk.LabelFrame(self.filter_frame, text="符合条件图片（点击跳转）", padding="5", style="Custom.TLabelframe")
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # 列表滚动条
        list_vscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        list_vscroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Listbox 支持 font 参数
        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=list_vscroll.set, selectbackground="#4a86e8", 
                                        selectforeground="white", font=self.font_manager.get_font("listbox"))
        self.image_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_vscroll.config(command=self.image_listbox.yview)

        # 绑定列表点击事件
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        self.update_filtered_images()

        return self.filter_frame

    def toggle_filter(self):
        """切换过滤器显示/隐藏"""
        self.filter_visible = not self.filter_visible
        if self.filter_visible:
            self.filter_frame.grid()
            self.update_score_distribution()  # 初始更新分布图
            self.update_filtered_images()     # 初始更新列表
        else:
            self.filter_frame.grid_remove()

    def on_bin_count_change(self, value):
        """评分区间数量改变事件"""
        self.bin_count = int(float(value))
        self.bin_label.config(text=str(self.bin_count))
        self.update_score_distribution()
        self.update_filtered_images() # 段数改变时更新列表

    def update_score_distribution(self):
        """更新评分分布图表，在清除轴后重新绘制竖线"""
        # 确保轴存在
        if self.ax is None:
            self.init_vlines()
            return

        # 保存当前过滤区间值（关键：在清除前保存状态）
        current_min = self.min_filter_score
        current_max = self.max_filter_score

        # 清除原有图表内容（会删除包括竖线在内的所有元素）
        self.ax.clear()

        # 获取有效评分数据并绘制直方图
        valid_scores = [score for score in self.parent.scores.values() if 0.0 <= score <= 9.0]
        if not valid_scores:
            self.ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", transform=self.ax.transAxes)
            self.ax.set_xlim(0, 9)
            self.ax.set_ylim(0, 1)
        else:
            counts, bins, patches = self.ax.hist(valid_scores, bins=self.bin_count, range=(0, 9), 
                                                color="#8ecae6", edgecolor="#219ebc", alpha=0.7)
            self.ax.set_xlabel("SCORE", fontsize=self.font_manager.get_font("chart_label")[1])
            self.ax.set_ylabel("COUNT", fontsize=self.font_manager.get_font("chart_label")[1])
            self.ax.set_xlim(0, 9)
            self.ax.grid(axis="y", alpha=0.3)

            # 高亮显示过滤区间内的柱形
            for patch, bin_left, bin_right in zip(patches, bins[:-1], bins[1:]):
                if not (bin_right < current_min or bin_left > current_max):
                    patch.set_facecolor("#219ebc")
                    patch.set_alpha(0.9)

            self.ax.axvspan(current_min, current_max, alpha=0.1, color="#e63946")

        # 关键修复：清除后重新创建竖线，并恢复之前的位置
        self.min_filter_score = current_min
        self.max_filter_score = current_max
        self.vline_left = None  # 重置线条引用
        self.vline_right = None
        self.init_vlines()  # 重新创建线条

        self.canvas.draw()

    def init_vlines(self):
        """初始化竖线，确保正确关联到图表"""
        if self.ax is None:
            self.figure, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.filter_frame)
            self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
            self.canvas.figure = self.figure
            self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 移除可能存在的旧线条
        if self.vline_left is not None:
            self.vline_left.remove()
        if self.vline_right is not None:
            self.vline_right.remove()

        # 创建新线条并关联到当前轴
        self.vline_left = self.ax.axvline(
            x=self.min_filter_score, color="#e63946", linewidth=2, linestyle="--", picker=5
        )
        self.vline_right = self.ax.axvline(
            x=self.max_filter_score, color="#e63946", linewidth=2, linestyle="--", picker=5
        )

        # 验证线条是否正确关联到图表（使用figure属性）
        assert self.vline_left.figure == self.figure, "竖线未正确关联到图表"
        assert self.vline_right.figure == self.figure, "竖线未正确关联到图表"

        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        """鼠标按下事件，使用正确的figure属性检查"""
        # 确保轴和线条存在
        if self.ax is None or self.vline_left is None or self.vline_right is None:
            self.init_vlines()
            return

        # 检查事件是否在当前图表内（使用正确的figure属性）
        if event.inaxes is None or event.inaxes.figure != self.figure:
            return

        # 检查线条是否属于当前图表（修正属性引用：figure而非fig）
        if (self.vline_left.figure != self.figure) or (self.vline_right.figure != self.figure):
            self.init_vlines()
            return

        # 处理点击事件
        if self.vline_left.contains(event)[0]:
            self.is_dragging_left = True
        elif self.vline_right.contains(event)[0]:
            self.is_dragging_right = True

    def on_mouse_release(self, event):
        """鼠标释放事件（结束拖拽竖线）"""
        self.is_dragging_left = False
        self.is_dragging_right = False

    def on_mouse_move(self, event):
        """鼠标移动事件（拖拽竖线）- 增加安全检查"""
        # 1. 先判断事件是否来自当前图表的轴
        if event.inaxes != self.ax:
            return

        # 2. 检查竖线是否已初始化（核心修复）
        if self.vline_left is None or self.vline_right is None:
            self.init_vlines()
            return

        # 3. 正常处理拖拽逻辑
        if self.is_dragging_left:
            new_x = max(0.0, min(event.xdata, self.max_filter_score - 0.1))
            self.min_filter_score = round(new_x, 1)
            self.vline_left.set_xdata(self.min_filter_score)
            # 拖拽时更新列表
            self.update_filtered_images()
        elif self.is_dragging_right:
            new_x = max(self.min_filter_score + 0.1, min(event.xdata, 9.0))
            self.max_filter_score = round(new_x, 1)
            self.vline_right.set_xdata(self.max_filter_score)
            # 拖拽时更新列表
            self.update_filtered_images()
        
        # 拖拽时也更新图表（以反映新的过滤区域）
        self.update_score_distribution()

    def update_filtered_images(self):
        """更新符合条件的图片列表"""
        self.filtered_images = []

        # 遍历所有图片，筛选评分在区间内的图片
        for img_path in self.parent.image_paths: # 使用主UI的图片列表
            rel_path = os.path.relpath(img_path, self.parent.folder_selector.image_folder) # 使用FolderSelector的路径
            # 检查是否有评分且评分在过滤区间内
            if rel_path in self.parent.scores:
                score = self.parent.scores[rel_path]
                if self.min_filter_score <= score <= self.max_filter_score:
                    # 存储（图片路径, 相对路径, 评分）
                    self.filtered_images.append((img_path, rel_path, score))

        # 更新列表框
        self.image_listbox.delete(0, tk.END)
        for idx, (_, rel_path, score) in enumerate(self.filtered_images, 1):
            # 显示格式：序号 - 相对路径 (评分)
            display_text = f"{idx:2d} - {os.path.basename(rel_path)} ({score:.1f})"
            self.image_listbox.insert(tk.END, display_text)

    def on_image_select(self, event):
        """图片列表选择事件（跳转显示选中图片）"""
        selected_idx = self.image_listbox.curselection()
        if not selected_idx:
            return

        # 获取选中的图片路径
        selected_img_info = self.filtered_images[selected_idx[0]]
        target_img_path = selected_img_info[0]

        # 查找目标图片在主列表中的索引
        try:
            target_index = self.parent.image_paths.index(target_img_path)
            # 更新主UI的当前图片索引并显示
            self.parent.current_index = target_index
            self.parent.display_current_image()
            self.parent.update_info()
        except ValueError:
            messagebox.showwarning("警告", "未找到选中的图片")

class ImageScorerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Scorer")
        self.root.geometry("1280x720") 

        # 字体管理器
        self.font_manager = FontManager()

        # 变量初始化
        self.image_folder = ""
        self.json_file = ""
        self.image_paths = []
        self.scores = {} # 在这里初始化
        self.current_index = 0
        self.auto_next_after_score = tk.BooleanVar(value=True)  # 打分后自动下一张
        self.zoom_to_fit = tk.BooleanVar(value=False)  # 图片放大到窗口

        # 初始化各组件
        self.info_var = tk.StringVar(value="请选择图片目录开始评分") # 创建info_var
        self.folder_selector = FolderSelector(self, self.font_manager)
        self.control_buttons = ControlButtons(self, self.font_manager, self.auto_next_after_score, self.zoom_to_fit, self.toggle_filter)
        self.image_viewer = ImageViewer(self, self.font_manager)
        self.score_var = tk.StringVar(value="未评分")
        self.score_buttons = ScoreButtons(self, self.font_manager, self.score_var, self.update_score)
        self.info_display = InfoDisplay(self, self.font_manager)
        self.shortcut_display = ShortcutDisplay(self, self.font_manager)
        self.filter_system = ScoreFilterSystem(self, self.font_manager)
        self.preprocessing_ui = DataPreprocessingUI(self, self.font_manager)
        # 创建UI
        self.create_ui()

        # 绑定全局快捷键
        self.bind_shortcuts()

    def create_ui(self):
        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))

        # 主框架
        self.main_frame = ttk.Frame(self.root, padding=self.font_manager.get_padding("main"))
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        # 创建各组件UI
        self.folder_selector.create_ui(self.main_frame, self.info_var) # 传递info_var
        self.control_buttons.create_ui(self.main_frame)
        self.image_viewer.create_ui(self.main_frame)
        self.score_buttons.create_ui(self.main_frame, self.toggle_filter) # 传递filter_toggle_func
        self.info_display.create_ui(self.main_frame)
        self.shortcut_display.create_ui(self.main_frame)
        self.filter_system.create_ui(self.main_frame)
        self.preprocessing_ui.create_ui(self.main_frame)

    def bind_shortcuts(self):
        """绑定全局快捷键"""
        # 数字键快速打分 (0-9)
        for i in range(10):
            self.root.bind(str(i), lambda event, x=i: self.score_buttons.set_score(x))

        # 方向键微调分数 (不自动下一张)
        self.root.bind('<Up>', lambda event: self.score_buttons.adjust_score(0.5, auto_next=False))
        self.root.bind('<Down>', lambda event: self.score_buttons.adjust_score(-0.5, auto_next=False))
        self.root.bind('<Right>', lambda event: self.score_buttons.adjust_score(0.1, auto_next=False))
        self.root.bind('<Left>', lambda event: self.score_buttons.adjust_score(-0.1, auto_next=False))

        # 导航快捷键
        self.root.bind('q', lambda event: self.previous_image())
        self.root.bind('e', lambda event: self.next_image())
        self.root.bind('w', lambda event: self.save_and_exit())
        self.root.bind('f', lambda event: self.toggle_filter())

        # 其他功能键
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.root.bind('<Return>', self.update_score)

        # 确保输入框也能响应Enter键
        self.score_buttons.score_entry.bind('<Return>', self.update_score)

    def get_all_image_paths(self):
        """获取文件夹及其子文件夹中的所有图片路径"""
        # 使用FolderSelector中的路径
        image_folder = self.folder_selector.image_folder
        if not image_folder:
            return []

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')
        image_paths = []

        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def load_images(self):
        """加载图片"""
        # 使用FolderSelector中的路径
        if not self.folder_selector.image_folder:
            messagebox.showwarning("警告", "请先选择图片目录")
            return

        self.image_paths = self.get_all_image_paths()
        if not self.image_paths:
            messagebox.showinfo("信息", "在指定目录中未找到图片文件")
            return

        self.current_index = 0
        # load_scores 通常在选择文件夹或JSON时已经调用过
        # 如果需要在加载图片时再次加载（例如外部修改了JSON），可以取消注释下一行
        # self.folder_selector.load_scores() 
        self.display_current_image()
        self.update_info()

        # 设置焦点到主窗口，确保快捷键生效
        self.root.focus_set()

    def display_current_image(self):
        """显示当前图片"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return

        image_path = self.image_paths[self.current_index]
        # 确保使用 FolderSelector 中的 image_folder 来计算相对路径
        rel_path = os.path.relpath(image_path, self.folder_selector.image_folder)

        self.image_viewer.display_image(image_path, self.zoom_to_fit.get())

        # 更新分数显示 - 检查图片是否已评分
        if rel_path in self.scores:
            current_score = self.scores[rel_path]
            self.score_var.set(f"{current_score:.1f}")
        else:
            self.score_var.set("未评分")
            
        # 也更新信息栏中的状态
        self.update_info() # 确保信息栏也反映了当前图片的评分状态

    def update_score(self, event=None):
        """更新当前图片的分数"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return

        try:
            score_text = self.score_var.get()
            if score_text == "未评分":
                raise ValueError("请先设置分数")

            score = float(score_text)
            if score < 0 or score > 9:  # 分数范围改为0-9
                raise ValueError("分数必须在0-9之间")

            image_path = self.image_paths[self.current_index]
            # 确保使用 FolderSelector 中的 image_folder 来计算相对路径
            rel_path = os.path.relpath(image_path, self.folder_selector.image_folder)
            self.scores[rel_path] = score

            # 自动保存
            self.folder_selector.save_scores()
            self.update_info()

            # 如果启用自动下一张，则跳转到下一张
            if self.auto_next_after_score.get():
                self.next_image()
            else:
                # 保持焦点在主窗口
                self.root.focus_set()

            # 更新过滤系统（重新计算分布和筛选）
            if self.filter_system.filter_visible:
                self.filter_system.update_score_distribution()
                self.filter_system.update_filtered_images()

        except ValueError as e:
            messagebox.showwarning("警告", f"无效的分数: {e}")
            # 恢复原分数显示
            image_path = self.image_paths[self.current_index]
            rel_path = os.path.relpath(image_path, self.folder_selector.image_folder)
            if rel_path in self.scores:
                current_score = self.scores[rel_path]
                self.score_var.set(f"{current_score:.1f}")
            else:
                self.score_var.set("未评分")

    def previous_image(self):
        """显示上一张图片"""
        if not self.image_paths:
            return

        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.display_current_image()
        self.update_info()

        # 设置焦点到主窗口
        self.root.focus_set()

    def next_image(self):
        """显示下一张图片"""
        if not self.image_paths:
            return

        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.display_current_image()
        self.update_info()

        # 设置焦点到主窗口
        self.root.focus_set()

    def update_info(self):
        """更新信息显示"""
        if not self.image_paths:
            return

        image_path = self.image_paths[self.current_index]
        rel_path = os.path.relpath(image_path, self.image_folder)
        filename = os.path.basename(image_path)

        scored_count = len([p for p in self.image_paths if os.path.relpath(p, self.image_folder) in self.scores])

        # 显示当前图片评分状态
        if rel_path in self.scores:
            score_status = f"已评分: {self.scores[rel_path]:.1f}"
        else:
            score_status = "未评分"

        info_text = f"当前图片: {filename} | {score_status}"
        progress_text = f"{self.current_index + 1}/{len(self.image_paths)} (已评分: {scored_count})"

        self.info_display.info_var.set(info_text)
        self.info_display.progress_var.set(progress_text)

    def toggle_filter(self):
        """切换过滤器显示/隐藏"""
        self.filter_system.toggle_filter()

    def save_and_exit(self):
        """保存并退出"""
        self.folder_selector.save_scores()
        #销毁过滤系统的图表资源（避免内存泄漏）
        if self.filter_system.figure:
            plt.close(self.filter_system.figure)
        self.root.quit()
        self.root.destroy()

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os

class DataPreprocessingUI(BaseComponent):
    """数据预处理功能UI组件"""
    def __init__(self, parent, font_manager):
        super().__init__(parent, font_manager)
        self.is_expanded = False
        self.preprocessing_frame = None
        self.data_folder_var = tk.StringVar(value=r"")
        self.output_folder_var = tk.StringVar(value=r"")
        self.augmentation_vars = {
            'resize': tk.BooleanVar(value=True),
            'flip': tk.BooleanVar(value=True),
            'rotation': tk.BooleanVar(value=False),
            'jitter': tk.BooleanVar(value=False),
            'noise': tk.BooleanVar(value=True),
        }
        self.batch_size_var = tk.IntVar(value=128)
        self.save_batch_size_var = tk.IntVar(value=10000)
        # 用于控制预处理线程
        self.preprocessing_thread = None
        self.stop_preprocessing = threading.Event()
        # 用于更新UI的队列
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()

    def create_ui(self, parent_frame):
        # 主容器框架（初始隐藏）
        self.preprocessing_frame = ttk.LabelFrame(parent_frame, text="数据预处理", padding="10")
        # 放在右侧，与过滤器不重叠，例如在过滤器下方或旁边
        # 为了与过滤器分开，我们放在过滤器旁边，但使用不同的row和column
        # 例如，如果过滤器在 column=3, 那么预处理可以放在 column=4 或者在下方的 row
        # 这里我们尝试放在一个新列
        self.preprocessing_frame.grid(row=2, column=4, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(0, 10))
        self.preprocessing_frame.columnconfigure(0, weight=1)
        self.preprocessing_frame.rowconfigure(5, weight=1) # 给日志区域分配更多权重
        self.preprocessing_frame.grid_remove()  # 初始隐藏

        # 配置 LabelFrame 样式以控制标题字体
        label_frame_style = ttk.Style()
        label_frame_style.configure("Custom.TLabelframe.Label", font=self.font_manager.get_font("title"))


        # 数据文件夹
        data_folder_frame = ttk.Frame(self.preprocessing_frame)
        data_folder_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(data_folder_frame, text="数据文件夹:", font=self.font_manager.get_font("label")).pack(side=tk.LEFT)
        ttk.Entry(data_folder_frame, textvariable=self.data_folder_var, font=self.font_manager.get_font("entry")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Button(data_folder_frame, text="浏览", command=self.select_data_folder, style="Custom.TButton").pack(side=tk.LEFT, padx=(5, 0))

        # 输出文件夹
        output_folder_frame = ttk.Frame(self.preprocessing_frame)
        output_folder_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(output_folder_frame, text="输出文件夹:", font=self.font_manager.get_font("label")).pack(side=tk.LEFT)
        ttk.Entry(output_folder_frame, textvariable=self.output_folder_var, font=self.font_manager.get_font("entry")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Button(output_folder_frame, text="浏览", command=self.select_output_folder, style="Custom.TButton").pack(side=tk.LEFT, padx=(5, 0))

        # 增强选项
        aug_frame = ttk.LabelFrame(self.preprocessing_frame, text="增强选项", padding="5")
        aug_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 配置 Checkbutton 样式
        checkbutton_style = ttk.Style()
        checkbutton_style.configure("Custom.TCheckbutton", font=self.font_manager.get_font("button"))
        
        for i, (aug_name, var) in enumerate(self.augmentation_vars.items()):
            ttk.Checkbutton(aug_frame, text=aug_name.capitalize(), variable=var, style="Custom.TCheckbutton").grid(row=i//2, column=i%2, sticky=tk.W, padx=(0, 10))

        # 批次大小
        batch_frame = ttk.Frame(self.preprocessing_frame)
        batch_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(batch_frame, text="批次大小:", font=self.font_manager.get_font("label")).pack(side=tk.LEFT)
        # 使用 Spinbox 而不是 Scale，更适合输入具体数值
        spinbox_style = ttk.Style()
        spinbox_style.configure("Custom.TSpinbox", font=self.font_manager.get_font("entry"))
        ttk.Spinbox(batch_frame, from_=1, to=1024, textvariable=self.batch_size_var, width=10, style="Custom.TSpinbox").pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(batch_frame, text="保存批次大小:", font=self.font_manager.get_font("label")).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Spinbox(batch_frame, from_=1000, to=50000, textvariable=self.save_batch_size_var, width=10, style="Custom.TSpinbox").pack(side=tk.LEFT, padx=(5, 0))

        # 控制按钮
        control_frame = ttk.Frame(self.preprocessing_frame)
        control_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Button(control_frame, text="开始预处理", command=self.start_preprocessing, style="Custom.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="停止预处理", command=self.stop_preprocessing_func, style="Custom.TButton").pack(side=tk.LEFT)
        ttk.Button(control_frame, text="展开/收起", command=self.toggle_expansion, style="Custom.TButton").pack(side=tk.RIGHT)

        # 日志显示区域
        log_frame = ttk.LabelFrame(self.preprocessing_frame, text="日志", padding="5")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.log_text = tk.Text(log_frame, yscrollcommand=log_scrollbar.set, state=tk.DISABLED, font=self.font_manager.get_font("listbox"))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.config(command=self.log_text.yview)

        # 配置按钮样式
        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=self.font_manager.get_font("button"))
        # 配置 Spinbox 样式已在上面完成
        # 配置 Checkbutton 样式已在上面完成

        return self.preprocessing_frame

    def select_data_folder(self):
        folder = filedialog.askdirectory(title="选择数据文件夹", initialdir=self.data_folder_var.get())
        if folder:
            self.data_folder_var.set(folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择输出文件夹", initialdir=self.output_folder_var.get())
        if folder:
            self.output_folder_var.set(folder)

    def start_preprocessing(self):
        """启动预处理线程"""
        if self.preprocessing_thread and self.preprocessing_thread.is_alive():
            messagebox.showwarning("警告", "预处理已在进行中！")
            return

        self.stop_preprocessing.clear() # 清除停止标志
        self.preprocessing_thread = threading.Thread(target=self.run_preprocessing, daemon=True)
        self.preprocessing_thread.start()
        
        # 启动UI更新定时器
        self.update_ui_loop()

    def update_ui_loop(self):
        """在主线程中定期检查队列并更新UI"""
        # 处理日志队列
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.append_log(message)
        except queue.Empty:
            pass
        
        # 处理进度队列
        try:
            while True:
                # 这里可以根据队列中的数据更新进度条或状态栏
                # 例如，更新一个进度标签
                processed_img, total_img, batch_size, total_proc, total_exp = self.progress_queue.get_nowait()
                # print(f"Progress: {processed_img}/{total_img}, Batch: {batch_size}, Total: {total_proc}/{total_exp}")
                # 可以在这里更新一个 ttk.Label 显示进度
        except queue.Empty:
            pass
            
        # 每隔一段时间（例如100毫秒）再次调用自己
        if self.preprocessing_thread and self.preprocessing_thread.is_alive():
            self.parent.root.after(100, self.update_ui_loop) # 100ms
    def append_log(self, message):
        """将信息添加到日志队列"""
        self.log_queue.put(message)
        
    # 或者，如果你不想用队列，可以直接更新UI，但需要确保在主线程中
    # def append_log(self, message):
    #     """在日志文本框中追加信息"""
    #     # 在主线程中更新UI
    #     self.log_text.config(state=tk.NORMAL)
    #     self.log_text.insert(tk.END, message + "\n")
    #     self.log_text.see(tk.END) # 滚动到底部
    #     self.log_text.config(state=tk.DISABLED)
    def run_preprocessing(self):
        """在新线程中运行预处理逻辑"""
        try:
            # 从UI获取参数
            root_folder = self.data_folder_var.get()
            output_folder = self.output_folder_var.get()
            augmentations = [name for name, var in self.augmentation_vars.items() if var.get()]
            batch_size = self.batch_size_var.get()
            save_batch_size = self.save_batch_size_var.get()

            """预处理数据并保存到磁盘，使用GPU加速和分块保存"""
            # 检查根文件夹是否存在
            if not os.path.exists(root_folder):
                self.append_log(f"Root folder does not exist: {root_folder}")
            
            # 创建保存路径
            os.makedirs(output_folder, exist_ok=True)
            
            # 加载所有数据
            image_paths, scores = self.load_all_data(root_folder)
            
            if len(image_paths) == 0:
                self.append_log("No images found or no matching scores in JSON files")
            
            # 随机打乱数据
            combined = list(zip(image_paths, scores))
            random.shuffle(combined)
            image_paths, scores = zip(*combined)
            
            # 创建增强配置
            augmentation_config = {}
            if 'resize' in augmentations:
                augmentation_config['resize'] = 2  # 2种缩放
            if 'flip' in augmentations:
                augmentation_config['flip'] = 2    # 翻转：是/否
            if 'rotation' in augmentations:
                augmentation_config['rotation'] = 3  # 3种旋转
            if 'jitter' in augmentations:
                augmentation_config['jitter'] = 2    # 颜色抖动：是/否
            if 'noise' in augmentations:
                augmentation_config['noise'] = 2     # 噪声：是/否

            # 创建增强生成器
            aug_generator = AugmentationGenerator(augmentation_config, device)
            num_combinations = len(aug_generator.augmentation_combinations)
            self.append_log(f"Generated {num_combinations} augmentation combinations")
            
            # 创建基础数据集
            dataset = BasicPreprocessDataset(image_paths, scores)
            
            # 使用单进程DataLoader以避免多进程问题
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,  # 保持顺序以便于索引
                num_workers=0,  # 改为0，使用单进程
                pin_memory=True
            )
            
            self.append_log(f"Preprocessing {len(dataset)} images with {num_combinations} augmentations each...")
            self.append_log(f"Total expected samples: {len(dataset) * num_combinations}")
            
            # 用于累积的列表
            all_images = []
            all_scores = []
            total_processed = 0
            chunk_count = 0
            
            # 创建分块目录
            images_chunk_dir = os.path.join(output_folder, 'images_chunks')
            scores_chunk_dir = os.path.join(output_folder, 'scores_chunks')
            os.makedirs(images_chunk_dir, exist_ok=True)
            os.makedirs(scores_chunk_dir, exist_ok=True)

            log_redirector = LogRedirector(self.append_log)

            # 使用tqdm创建进度条
            pbar = tqdm(total=len(dataset), desc="预处理图片", unit="images", file=log_redirector)
            
            for batch_idx, (images, scores_batch) in enumerate(dataloader):
                images = images.to(device)
                scores_batch = scores_batch.to(device)
                
                # 应用增强
                augmented_images, augmented_scores = aug_generator.apply_augmentation_to_batch(images, scores_batch)
                
                # 添加到累积列表
                all_images.append(augmented_images.cpu())
                all_scores.append(augmented_scores.cpu())
                
                current_batch_size = augmented_images.size(0)
                total_processed += current_batch_size
                
                # 更新进度条
                pbar.set_postfix({
                    'Processed': f'{total_processed}/{len(dataset) * num_combinations}',
                    'Batch Size': current_batch_size
                })
                pbar.update(images.size(0))
                
                # 当累积的数据达到保存批次大小时，保存为分块文件
                if total_processed >= save_batch_size:
                    # 合并当前累积的数据
                    current_images = torch.cat(all_images, dim=0)
                    current_scores = torch.cat(all_scores, dim=0)
                    
                    # 保存图像和分数分块
                    images_chunk_path = os.path.join(images_chunk_dir, f'images_chunk_{chunk_count:04d}.pt')
                    scores_chunk_path = os.path.join(scores_chunk_dir, f'scores_chunk_{chunk_count:04d}.pt')
                    
                    torch.save(current_images, images_chunk_path)
                    torch.save(current_scores, scores_chunk_path)
                    
                    self.append_log(f"Saved chunk {chunk_count} - Images: {current_images.size(0)}, Scores: {current_scores.size(0)}")
                    
                    # 清空累积列表
                    all_images = []
                    all_scores = []
                    chunk_count += 1
            
            # 处理最后剩余的数据
            if all_images:
                current_images = torch.cat(all_images, dim=0)
                current_scores = torch.cat(all_scores, dim=0)
                
                # 保存最后一个分块
                images_chunk_path = os.path.join(images_chunk_dir, f'images_chunk_{chunk_count:04d}.pt')
                scores_chunk_path = os.path.join(scores_chunk_dir, f'scores_chunk_{chunk_count:04d}.pt')
                
                torch.save(current_images, images_chunk_path)
                torch.save(current_scores, scores_chunk_path)
                
                self.append_log(f"Saved final chunk {chunk_count} - Images: {current_images.size(0)}, Scores: {current_scores.size(0)}")
                chunk_count += 1
            
            pbar.close()
            
            # 保存元数据
            metadata = {
                'num_chunks': chunk_count,
                'num_combinations': num_combinations,
                'original_dataset_size': len(dataset),
                'total_samples': total_processed
            }
            
            metadata_path = os.path.join(output_folder, 'metadata.json')
            import json as js
            with open(metadata_path, 'w') as f:
                js.dump(metadata, f)
            
            self.append_log(f"Preprocessing completed!")
            self.append_log(f"Saved {chunk_count} chunks to {images_chunk_dir} and {scores_chunk_dir}")
            self.append_log(f"Total samples processed: {total_processed}")
            pbar.close()
        except Exception as e:
            self.append_log(f"预处理出错: {e}")

    def stop_preprocessing_func(self):
        """设置停止标志"""
        self.stop_preprocessing.set()
        self.append_log("正在请求停止预处理...")

    def append_log(self, message):
        """在日志文本框中追加信息"""
        # 在主线程中更新UI
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # 滚动到底部
        self.log_text.config(state=tk.DISABLED)

    def toggle_expansion(self):
        """切换展开/收起"""
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.preprocessing_frame.grid()

            # 设置默认路径：使用主UI的图片目录
            if self.data_folder_var.get() or self.output_folder_var.get():
                return  # 如果已经设置了路径，就不覆盖
            initial_data_path = self.parent.folder_selector.image_folder
            initial_output_path = os.path.join(initial_data_path, "preprocessed_data") if initial_data_path else r""
            
            self.data_folder_var.set(initial_data_path)
            self.output_folder_var.set(initial_output_path)
        else:
            self.preprocessing_frame.grid_remove()

    def expand(self):
        """展开面板"""
        self.is_expanded = True
        self.preprocessing_frame.grid()

    def collapse(self):
        """收起面板"""
        self.is_expanded = False
        self.preprocessing_frame.grid_remove()
    # 检查并加载单个文件夹中的数据
    def load_single_folder_data(self, folder_path):
        json_path = os.path.join(folder_path, 'scores.json')
        if not os.path.exists(json_path):
            return [], []
        
        with open(json_path, 'r') as f:
            score_dict = json.load(f)

        image_paths = []
        scores = []
        
        # 只检查当前文件夹，不递归到子文件夹
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if file in score_dict:
                    image_path = os.path.join(folder_path, file)
                    image_paths.append(image_path)
                    scores.append(score_dict[file])
        
        # 验证所有JSON中描述的文件是否都存在
        missing_files = []
        for file in score_dict.keys():
            if not os.path.exists(os.path.join(folder_path, file)):
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"In folder {folder_path}, the following files are referenced in scores.json but not found: {missing_files}")
        
        print(f"Loaded {len(image_paths)} images from {folder_path}")
        return image_paths, scores

    # 递归遍历所有子文件夹并收集数据
    def load_all_data(self, root_folder):
        all_image_paths = []
        all_scores = []
        
        # 遍历所有子文件夹
        for root, dirs, files in os.walk(root_folder):
            # 检查当前文件夹是否包含scores.json
            if 'scores.json' in files:
                try:
                    img_paths, scores = self.load_single_folder_data(root)
                    all_image_paths.extend(img_paths)
                    all_scores.extend(scores)
                except Exception as e:
                    print(f"Error processing folder {root}: {e}")
                    continue
        
        if len(all_image_paths) == 0:
            raise ValueError("No valid data found in any subfolder")
        
        print(f"Total loaded {len(all_image_paths)} images from all subfolders")
        return all_image_paths, all_scores

class LogRedirector:
    """将输出重定向到UI日志框的类"""
    def __init__(self, append_log_func):
        self.append_log_func = append_log_func
        self.buffer = ""

    def write(self, message):
        # tqdm 可能会输出包含换行符的多行消息
        # 我们需要按行处理
        self.buffer += message
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # 最后一行可能不完整，留到下次处理
            self.buffer = lines[-1]
            # 处理完整的行
            for line in lines[:-1]:
                # tqdm 的进度条通常包含回车符 \r，用于覆盖本行
                # 我们只取 \r 之后的内容，即最终显示的行
                final_line = line.split('\r')[-1]
                if final_line.strip(): # 忽略空行
                    self.append_log_func(final_line)

    def flush(self):
        # tqdm 有时会调用 flush，我们在这里处理剩余的缓冲区
        if self.buffer.strip():
            final_line = self.buffer.split('\r')[-1]
            if final_line.strip():
                self.append_log_func(final_line)
        self.buffer = ""



def main():
    root = tk.Tk()
    app = ImageScorerUI(root)
    root.mainloop()

# GPU加速的增强操作
class GPUAugmentationPipeline:
    def __init__(self, device):
        self.device = device
        # GPU加速的增强操作，优化颜色抖动参数
        self.color_jitter = K.ColorJitter(
            brightness=0.1,  # 降低亮度抖动幅度
            contrast=0.1,   # 降低对比度抖动幅度
            saturation=0.1, # 降低饱和度抖动幅度
            hue=0.05,       # 降低色调抖动幅度
            p=1.0           # 应用概率设置为1.0，由组合控制是否应用
        ).to(device)
        self.flip_horizontal = K.RandomHorizontalFlip(p=1.0).to(device)
        self.flip_vertical = K.RandomVerticalFlip(p=1.0).to(device)
        # 使用兼容的 RandomRotation 参数（移除 fill）
        self.rotation = K.RandomRotation(
            degrees=30.0,
            p=1.0
        ).to(device)

    def apply_jitter(self, image_tensor):
        # 应用颜色抖动并确保像素值在合理范围内
        jittered = self.color_jitter(image_tensor)
        return torch.clamp(jittered, 0, 1)  # 限制像素值在 [0, 1]

    def apply_flip_horizontal(self, image_tensor):
        return self.flip_horizontal(image_tensor)

    def apply_flip_vertical(self, image_tensor):
        return self.flip_vertical(image_tensor)

    def apply_rotation(self, image_tensor):
        # 应用旋转
        rotated = self.rotation(image_tensor)
        # 创建掩码以识别填充区域（假设默认填充为0）
        mask = (rotated == 0).all(dim=1, keepdim=True).float()  # 填充区域为0
        # 将填充区域设置为白色 (1.0)
        white = torch.ones_like(image_tensor) * 1.0
        rotated = (1 - mask) * rotated + mask * white
        return torch.clamp(rotated, 0, 1)  # 确保范围

    def add_noise(self, image_tensor, noise_std=0.05):
        # 生成高斯噪声，保护白色背景
        noise = torch.randn_like(image_tensor, device=self.device) * noise_std
        # 创建掩码，保护白色背景（接近 [1, 1, 1] 的像素）
        mask = (image_tensor >= 0.99).all(dim=1, keepdim=True).float()
        noisy_image = image_tensor + noise * (1 - mask)  # 只对非白色区域加噪声
        return torch.clamp(noisy_image, 0, 1)
    
# 用于生成增强的辅助类
class AugmentationGenerator:
    def __init__(self, augmentation_config, device):
        self.augmentation_config = augmentation_config
        self.device = device
        self.gpu_aug = GPUAugmentationPipeline(device)
        # 定义 mean 和 std 用于反归一化
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self._generate_augmentation_combinations()

    def _generate_augmentation_combinations(self):
        """生成所有可能的增强组合"""
        # 为每种增强生成选项
        augmentation_options = {}
        
        # 缩放选项
        if 'resize' in self.augmentation_config:
            resize_count = self.augmentation_config['resize']
            augmentation_options['resize'] = [
                {'scale': (0.8 + i * 0.1, 1.0)} for i in range(resize_count)
            ]
        
        # 翻转选项 - 修改为4种翻转类型
        if 'flip' in self.augmentation_config:
            flip_count = self.augmentation_config['flip']
            # 0: 无翻转, 1: 水平翻转, 2: 垂直翻转, 3: 水平垂直翻转
            augmentation_options['flip'] = [
                {'type': i} for i in range(flip_count)
            ]
        
        # 旋转选项
        if 'rotation' in self.augmentation_config:
            rotation_count = self.augmentation_config['rotation']
            # 使用正的角度范围，RandomRotation内部会处理正负
            augmentation_options['rotation'] = [
                {'degrees': 10 * (i + 1)} for i in range(rotation_count)  # 10, 20, 30度
            ]
        
        # 颜色抖动选项
        if 'jitter' in self.augmentation_config:
            jitter_count = self.augmentation_config['jitter']
            augmentation_options['jitter'] = [
                {'apply': bool(i)} for i in range(jitter_count)
            ]
        
        # 噪声选项
        if 'noise' in self.augmentation_config:
            noise_count = self.augmentation_config['noise']
            augmentation_options['noise'] = [
                {'apply': bool(i)} for i in range(noise_count)
            ]
        
        # 生成所有组合
        from itertools import product
        keys = list(augmentation_options.keys())
        values = [augmentation_options[key] for key in keys]
        
        combinations = []
        for combination in product(*values):
            combo_dict = {}
            for key, value in zip(keys, combination):
                combo_dict[key] = value
            combinations.append(combo_dict)
        
        self.augmentation_combinations = combinations

    def apply_augmentation_to_batch(self, images_batch, scores_batch):
        """对一批图像应用所有增强组合"""
        augmented_images = []
        augmented_scores = []
        
        batch_size = images_batch.size(0)
        num_combinations = len(self.augmentation_combinations)
        
        for combo_idx, augmentation_combo in enumerate(self.augmentation_combinations):
            augmented_batch = images_batch.clone()  # 复制基础图像
            
            # 反归一化到 [0, 1] 范围
            augmented_batch = (augmented_batch * self.std) + self.mean
            augmented_batch = torch.clamp(augmented_batch, 0, 1)
            
            # 应用增强到整个批次（在 [0,1] 空间）
            for i in range(batch_size):
                img = augmented_batch[i:i+1]  # 取单个图像，形状为 [1, C, H, W]
                
                # 应用各种增强（GPU加速）
                if 'jitter' in augmentation_combo and augmentation_combo['jitter']['apply']:
                    img = self.gpu_aug.apply_jitter(img)
                
                if 'flip' in augmentation_combo:
                    flip_type = augmentation_combo['flip']['type']
                    if flip_type == 1:  # 水平翻转
                        img = self.gpu_aug.apply_flip_horizontal(img)
                    elif flip_type == 2:  # 垂直翻转
                        img = self.gpu_aug.apply_flip_vertical(img)
                    elif flip_type == 3:  # 水平垂直翻转
                        img = self.gpu_aug.apply_flip_horizontal(img)
                        img = self.gpu_aug.apply_flip_vertical(img)
                
                if 'rotation' in augmentation_combo:
                    degrees = augmentation_combo['rotation']['degrees']
                    # 动态设置旋转角度（因为组合有不同degrees）
                    self.gpu_aug.rotation.degrees = degrees
                    img = self.gpu_aug.apply_rotation(img)
                
                if 'noise' in augmentation_combo and augmentation_combo['noise']['apply']:
                    img = self.gpu_aug.add_noise(img, noise_std=0.05)
                
                augmented_batch[i:i+1] = img  # 将增强后的图像放回批次
            
            # 重新归一化
            augmented_batch = (augmented_batch - self.mean) / self.std
            
            augmented_images.append(augmented_batch)
            # 复制分数批次
            augmented_scores.append(scores_batch)
        
        # 合并所有增强批次
        final_images = torch.cat(augmented_images, dim=0)  # [batch_size * num_combinations, C, H, W]
        final_scores = torch.cat(augmented_scores, dim=0)  # [batch_size * num_combinations]
        
        return final_images, final_scores
# 自定义图片缩放变换 - 保持宽高比，填充而不是裁切
class ResizeWithPad:
    def __init__(self, target_size):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, img):
        # 获取原始图片尺寸
        w, h = img.size
        target_w, target_h = self.target_size
        
        # 计算缩放比例，保持宽高比
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图片
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # 计算填充
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        
        # 填充图片（使用白色填充）
        img = transforms.Pad(padding, fill=(255, 255, 255))(img)
        
        return img

# 自定义随机缩放裁切变换 - 保持宽高比
class RandomResizeWithPad:
    def __init__(self, target_size, scale=(0.8, 1.0)):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.scale = scale

    def __call__(self, img):
        # 随机选择缩放比例
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        
        # 获取原始图片尺寸
        w, h = img.size
        target_w, target_h = self.target_size
        
        # 计算缩放后的尺寸
        scaled_w = int(w * scale_factor)
        scaled_h = int(h * scale_factor)
        
        # 按比例缩放图片
        img = img.resize((scaled_w, scaled_h), Image.BILINEAR)
        
        # 计算填充
        delta_w = target_w - scaled_w
        delta_h = target_h - scaled_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        
        # 填充图片（使用白色填充）
        img = transforms.Pad(padding, fill=(255, 255, 255))(img)
        
        return img

# 简化版的正交化增强数据集（不包含增强，只做基础预处理）
class BasicPreprocessDataset(Dataset):
    def __init__(self, image_paths, scores):
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transforms.Compose([
            ResizeWithPad(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 修复EXIF问题
            image = Image.open(img_path).convert("RGB")
            # 尝试修复EXIF数据
            image.load()  # 这会强制加载图像数据
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        score = self.scores[idx]
        image_tensor = self.transform(image)
        
        return image_tensor, torch.tensor(score, dtype=torch.float32)

if __name__ == "__main__":
    main()