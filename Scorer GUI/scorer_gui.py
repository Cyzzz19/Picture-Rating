import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageScorerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Scorer")
        self.root.geometry("1200x800")
        
        # 变量初始化
        self.image_folder = ""
        self.json_file = ""
        self.image_paths = []
        self.scores = {}
        self.current_index = 0
        self.auto_next_after_score = tk.BooleanVar(value=True)  # 打分后自动下一张
        self.zoom_to_fit = tk.BooleanVar(value=False)  # 图片放大到窗口
        
        # 创建UI
        self.create_ui()
        
        # 绑定全局快捷键
        self.bind_shortcuts()
        
    def create_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 文件夹选择区域
        folder_frame = ttk.LabelFrame(main_frame, text="文件夹设置", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="图片目录:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_var, state='readonly').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(folder_frame, text="浏览", command=self.select_folder).grid(row=0, column=2)
        
        ttk.Label(folder_frame, text="JSON文件:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.json_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.json_var, state='readonly').grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(folder_frame, text="浏览", command=self.select_json).grid(row=1, column=2)
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="加载图片", command=self.load_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="上一张 (Q)", command=self.previous_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="下一张 (E)", command=self.next_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="保存并退出 (W)", command=self.save_and_exit).pack(side=tk.LEFT, padx=(0, 5))
        
        # 选项区域
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Checkbutton(options_frame, text="打分后自动下一张", 
                       variable=self.auto_next_after_score).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(options_frame, text="图片放大到窗口", 
                       variable=self.zoom_to_fit, command=self.display_current_image).pack(side=tk.LEFT)
        
        # 图片显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图片显示", padding="5")
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
        
        # 评分区域
        score_frame = ttk.LabelFrame(main_frame, text="评分", padding="5")
        score_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(score_frame, text="当前分数:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.score_var = tk.StringVar(value="未评分")
        self.score_entry = ttk.Entry(score_frame, textvariable=self.score_var, width=10)
        self.score_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        self.score_entry.bind('<Return>', self.update_score)
        
        # 快速评分按钮 (0-9，没有10)
        quick_score_frame = ttk.Frame(score_frame)
        quick_score_frame.grid(row=0, column=2, sticky=tk.W)
        
        for i in range(10):  # 0-9
            ttk.Button(quick_score_frame, text=str(i), 
                      command=lambda x=i: self.set_score(x)).pack(side=tk.LEFT, padx=2)
        
        # 微调分数按钮
        adjust_frame = ttk.Frame(score_frame)
        adjust_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        ttk.Button(adjust_frame, text="+0.5 (↑)", 
                  command=lambda: self.adjust_score(0.5, auto_next=False)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(adjust_frame, text="-0.5 (↓)", 
                  command=lambda: self.adjust_score(-0.5, auto_next=False)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(adjust_frame, text="+0.1 (→)", 
                  command=lambda: self.adjust_score(0.1, auto_next=False)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(adjust_frame, text="-0.1 (←)", 
                  command=lambda: self.adjust_score(-0.1, auto_next=False)).pack(side=tk.LEFT)
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="信息", padding="5")
        info_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(0, weight=1)
        
        self.info_var = tk.StringVar(value="请选择图片目录开始评分")
        ttk.Label(info_frame, textvariable=self.info_var).grid(row=0, column=0, sticky=tk.W)
        
        # 进度显示
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(info_frame, textvariable=self.progress_var).grid(row=0, column=1, sticky=tk.E)
        
        # 快捷键提示
        shortcut_frame = ttk.LabelFrame(main_frame, text="快捷键提示", padding="5")
        shortcut_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        shortcuts = [
            "Q: 上一张图片 | E: 下一张图片 | W: 保存并退出",
            "0-9: 快速打分 | ↑/↓: ±0.5分 | ←/→: ±0.1分",
            "Enter: 确认分数 | ESC: 退出程序"
        ]
        
        for i, shortcut in enumerate(shortcuts):
            ttk.Label(shortcut_frame, text=shortcut).grid(row=i, column=0, sticky=tk.W)
    
    def bind_shortcuts(self):
        """绑定全局快捷键"""
        # 数字键快速打分 (0-9)
        for i in range(10):
            self.root.bind(str(i), lambda event, x=i: self.set_score(x))
        
        # 方向键微调分数 (不自动下一张)
        self.root.bind('<Up>', lambda event: self.adjust_score(0.5, auto_next=False))
        self.root.bind('<Down>', lambda event: self.adjust_score(-0.5, auto_next=False))
        self.root.bind('<Right>', lambda event: self.adjust_score(0.1, auto_next=False))
        self.root.bind('<Left>', lambda event: self.adjust_score(-0.1, auto_next=False))
        
        # 导航快捷键
        self.root.bind('q', lambda event: self.previous_image())
        self.root.bind('e', lambda event: self.next_image())
        self.root.bind('w', lambda event: self.save_and_exit())
        
        # 其他功能键
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.root.bind('<Return>', self.update_score)
        
        # 确保输入框也能响应Enter键
        self.score_entry.bind('<Return>', self.update_score)
    
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
            self.scores = {}
            self.save_scores()
    
    def load_scores(self):
        """加载评分数据"""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self.scores = json.load(f)
                self.info_var.set(f"已加载 {len(self.scores)} 个评分")
            except Exception as e:
                messagebox.showerror("错误", f"加载JSON文件失败: {e}")
                self.scores = {}
        else:
            self.scores = {}
            self.save_scores()
    
    def save_scores(self):
        """保存评分数据"""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.scores, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("错误", f"保存JSON文件失败: {e}")
    
    def get_all_image_paths(self):
        """获取文件夹及其子文件夹中的所有图片路径"""
        if not self.image_folder:
            return []
            
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')
        image_paths = []
        
        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)
    
    def load_images(self):
        """加载图片"""
        if not self.image_folder:
            messagebox.showwarning("警告", "请先选择图片目录")
            return
            
        self.image_paths = self.get_all_image_paths()
        if not self.image_paths:
            messagebox.showinfo("信息", "在指定目录中未找到图片文件")
            return
            
        self.current_index = 0
        self.load_scores()
        self.display_current_image()
        self.update_info()
        
        # 设置焦点到主窗口，确保快捷键生效
        self.root.focus_set()
    
    def display_current_image(self):
        """显示当前图片"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
            
        image_path = self.image_paths[self.current_index]
        rel_path = os.path.relpath(image_path, self.image_folder)
        
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
            
            if self.zoom_to_fit.get():
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
            
            # 更新分数显示 - 检查图片是否已评分
            if rel_path in self.scores:
                current_score = self.scores[rel_path]
                self.score_var.set(f"{current_score:.1f}")
            else:
                self.score_var.set("未评分")
            
        except Exception as e:
            messagebox.showerror("错误", f"显示图片失败: {e}")
    
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
            rel_path = os.path.relpath(image_path, self.image_folder)
            self.scores[rel_path] = score
            
            # 自动保存
            self.save_scores()
            self.update_info()
            
            # 如果启用自动下一张，则跳转到下一张
            if self.auto_next_after_score.get():
                self.next_image()
            else:
                # 保持焦点在主窗口
                self.root.focus_set()
            
        except ValueError as e:
            messagebox.showwarning("警告", f"无效的分数: {e}")
            # 恢复原分数显示
            image_path = self.image_paths[self.current_index]
            rel_path = os.path.relpath(image_path, self.image_folder)
            if rel_path in self.scores:
                current_score = self.scores[rel_path]
                self.score_var.set(f"{current_score:.1f}")
            else:
                self.score_var.set("未评分")
    
    def set_score(self, score):
        """快速设置分数"""
        self.score_var.set(str(score))
        self.update_score()
    
    def adjust_score(self, delta, auto_next=True):
        """调整当前分数"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
            
        try:
            current_text = self.score_var.get()
            if current_text == "未评分":
                current_score = 0.0
            else:
                current_score = float(current_text)
            
            new_score = max(0.0, min(9.0, current_score + delta))  # 上限改为9.0
            self.score_var.set(f"{new_score:.1f}")
            
            # 立即更新分数到JSON
            image_path = self.image_paths[self.current_index]
            rel_path = os.path.relpath(image_path, self.image_folder)
            self.scores[rel_path] = new_score
            self.save_scores()
            
            # 只有明确要求auto_next时才自动下一张
            if auto_next and self.auto_next_after_score.get():
                self.next_image()
            else:
                self.root.focus_set()
                
        except ValueError:
            pass
    
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
        
        self.info_var.set(info_text)
        self.progress_var.set(progress_text)
    
    def save_and_exit(self):
        """保存并退出"""
        self.save_scores()
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ImageScorerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()