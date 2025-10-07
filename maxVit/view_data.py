import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as patches
import argparse
import os
import sys
import json

class ChunkedImageViewer:
    def __init__(self, data_folder, batch_size=1000):
        """
        初始化分块图片查看器
        :param data_folder: 预处理数据文件夹路径
        :param batch_size: 每次加载的批次大小
        """
        self.data_folder = data_folder
        self.batch_size = batch_size
        
        # 加载元数据
        metadata_path = os.path.join(data_folder, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_chunks = self.metadata['num_chunks']
        self.total_samples = self.metadata['total_samples']
        self.num_combinations = self.metadata['num_combinations']
        self.original_dataset_size = self.metadata['original_dataset_size']
        
        print(f"Total samples: {self.total_samples}")
        print(f"Number of chunks: {self.num_chunks}")
        print(f"Augmentation combinations: {self.num_combinations}")
        
        # 初始化显示参数
        self.current_idx = 0
        self.current_chunk = None
        self.current_chunk_scores = None
        self.current_chunk_id = -1
        self.current_chunk_start_idx = -1
        self.current_chunk_end_idx = -1
        
        # 计算每个chunk的大小
        self.chunk_sizes = self._calculate_chunk_sizes()
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Chunked Image Viewer')
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 创建按钮和控件
        self.create_controls()
        
        # 显示第一张图片
        self.update_display()
        
    def _calculate_chunk_sizes(self):
        """计算每个chunk的大小"""
        chunk_sizes = []
        images_chunk_dir = os.path.join(self.data_folder, 'images_chunks')
        
        for chunk_id in range(self.num_chunks):
            chunk_path = os.path.join(images_chunk_dir, f'images_chunk_{chunk_id:04d}.pt')
            if os.path.exists(chunk_path):
                # 只加载元数据来获取大小，不加载实际数据
                chunk_data = torch.load(chunk_path, map_location='cpu')
                chunk_sizes.append(len(chunk_data))
                del chunk_data
            else:
                chunk_sizes.append(0)
        
        return chunk_sizes
    
    def load_chunk(self, chunk_id):
        """加载指定的chunk"""
        images_chunk_path = os.path.join(self.data_folder, 'images_chunks', f'images_chunk_{chunk_id:04d}.pt')
        scores_chunk_path = os.path.join(self.data_folder, 'scores_chunks', f'scores_chunk_{chunk_id:04d}.pt')
        
        if not os.path.exists(images_chunk_path) or not os.path.exists(scores_chunk_path):
            raise FileNotFoundError(f"Chunk {chunk_id} not found")
        
        print(f"Loading chunk {chunk_id}...")
        images = torch.load(images_chunk_path, map_location='cpu')
        scores = torch.load(scores_chunk_path, map_location='cpu')
        
        # 计算chunk的全局索引范围
        start_idx = sum(self.chunk_sizes[:chunk_id])
        end_idx = start_idx + len(images)
        
        return images, scores, chunk_id, start_idx, end_idx
    
    def find_chunk_for_index(self, global_idx):
        """找到包含指定全局索引的chunk"""
        cumulative_size = 0
        for chunk_id, chunk_size in enumerate(self.chunk_sizes):
            if cumulative_size <= global_idx < cumulative_size + chunk_size:
                return chunk_id, cumulative_size
            cumulative_size += chunk_size
        return None, None
    
    def get_current_image_and_score(self):
        """获取当前索引的图片和分数"""
        # 找到包含当前索引的chunk
        chunk_id, chunk_start_idx = self.find_chunk_for_index(self.current_idx)
        
        if chunk_id is None:
            raise IndexError(f"Index {self.current_idx} out of range")
        
        # 如果当前chunk不是所需的chunk，则加载新的chunk
        if self.current_chunk_id != chunk_id:
            self.current_chunk, self.current_chunk_scores, self.current_chunk_id, self.current_chunk_start_idx, self.current_chunk_end_idx = self.load_chunk(chunk_id)
        
        # 计算在chunk内的局部索引
        local_idx = self.current_idx - self.current_chunk_start_idx
        current_image = self.current_chunk[local_idx]
        current_score = self.current_chunk_scores[local_idx].item()
        
        return current_image, current_score
    
    def update_display(self):
        """更新显示"""
        if self.current_idx >= self.total_samples:
            self.current_idx = self.total_samples - 1
        elif self.current_idx < 0:
            self.current_idx = 0
            
        try:
            # 获取当前图片和分数
            image_tensor, score = self.get_current_image_and_score()
            
            # 转换张量为可显示图像
            # 反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = image_tensor * std + mean
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # 转换为numpy并移动通道维度
            image_np = image_tensor.permute(1, 2, 0).numpy()
            
            # 清除当前轴
            self.ax.clear()
            
            # 显示图片
            self.ax.imshow(image_np)
            self.ax.axis('off')
            
            # 计算原始图像索引和增强类型
            original_idx = self.current_idx // self.num_combinations
            aug_type = self.current_idx % self.num_combinations
            
            # 添加标题和信息
            title = f'Image {self.current_idx + 1}/{self.total_samples} | Score: {score:.4f}'
            self.ax.set_title(title, fontsize=14, pad=20)
            
            # 添加详细信息
            info_text = f'Global Index: {self.current_idx}\nOriginal Index: {original_idx}\nAugmentation: {aug_type + 1}/{self.num_combinations}\nChunk: {self.current_chunk_id}/{self.num_chunks}'
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"Error displaying image {self.current_idx}: {e}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error loading image {self.current_idx}", 
                        transform=self.ax.transAxes, ha='center', va='center')
            self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 'right' or event.key == 'l':  # 右箭头或L键
            self.current_idx += 1
            self.update_display()
        elif event.key == 'left' or event.key == 'h':  # 左箭头或H键
            self.current_idx -= 1
            self.update_display()
        elif event.key == 'up' or event.key == 'k':  # 上箭头或K键 - 跳转到下一个原始图像
            self.current_idx = min(self.total_samples - 1, self.current_idx + self.num_combinations)
            self.update_display()
        elif event.key == 'down' or event.key == 'j':  # 下箭头或J键 - 跳转到上一个原始图像
            self.current_idx = max(0, self.current_idx - self.num_combinations)
            self.update_display()
        elif event.key == 'q' or event.key == 'escape':  # Q键或ESC键退出
            plt.close(self.fig)
        elif event.key == 'home':  # Home键跳到第一张
            self.current_idx = 0
            self.update_display()
        elif event.key == 'end':  # End键跳到最后一张
            self.current_idx = self.total_samples - 1
            self.update_display()
        elif event.key == 'pageup':  # PageUp键向前跳100张
            self.current_idx = max(0, self.current_idx - 100)
            self.update_display()
        elif event.key == 'pagedown':  # PageDown键向后跳100张
            self.current_idx = min(self.total_samples - 1, self.current_idx + 100)
            self.update_display()
        elif event.key == 'c':  # C键显示chunk信息
            self.show_chunk_info()
    
    def show_chunk_info(self):
        """显示当前chunk信息"""
        chunk_info = (f"Current Chunk: {self.current_chunk_id}\n"
                     f"Chunk Range: {self.current_chunk_start_idx} - {self.current_chunk_end_idx - 1}\n"
                     f"Chunk Size: {len(self.current_chunk) if self.current_chunk is not None else 0}\n"
                     f"Loaded Chunks in Memory: {1 if self.current_chunk is not None else 0}")
        print(chunk_info)
    
    def on_prev_click(self, event):
        """上一张按钮点击事件"""
        self.current_idx -= 1
        self.update_display()
    
    def on_next_click(self, event):
        """下一张按钮点击事件"""
        self.current_idx += 1
        self.update_display()
    
    def on_prev_original_click(self, event):
        """上一个原始图像点击事件"""
        self.current_idx = max(0, self.current_idx - self.num_combinations)
        self.update_display()
    
    def on_next_original_click(self, event):
        """下一个原始图像点击事件"""
        self.current_idx = min(self.total_samples - 1, self.current_idx + self.num_combinations)
        self.update_display()
    
    def on_jump_click(self, text):
        """跳转按钮点击事件"""
        try:
            target_idx = int(text)
            if 0 <= target_idx < self.total_samples:
                self.current_idx = target_idx
                self.update_display()
            else:
                print(f"Index must be between 0 and {self.total_samples - 1}")
        except ValueError:
            print("Please enter a valid number")
    
    def create_controls(self):
        """创建控制按钮和控件"""
        # 添加上一张按钮
        ax_prev = self.fig.add_axes([0.3, 0.01, 0.08, 0.04])
        btn_prev = Button(ax_prev, 'Prev (←)')
        btn_prev.on_clicked(self.on_prev_click)
        
        # 添加下一张按钮
        ax_next = self.fig.add_axes([0.4, 0.01, 0.08, 0.04])
        btn_next = Button(ax_next, 'Next (→)')
        btn_next.on_clicked(self.on_next_click)
        
        # 添加上一个原始图像按钮
        ax_prev_orig = self.fig.add_axes([0.5, 0.01, 0.1, 0.04])
        btn_prev_orig = Button(ax_prev_orig, 'Prev Original (↑)')
        btn_prev_orig.on_clicked(self.on_prev_original_click)
        
        # 添加下一个原始图像按钮
        ax_next_orig = self.fig.add_axes([0.62, 0.01, 0.1, 0.04])
        btn_next_orig = Button(ax_next_orig, 'Next Original (↓)')
        btn_next_orig.on_clicked(self.on_next_original_click)
        
        # 添加跳转控件
        ax_jump = self.fig.add_axes([0.74, 0.01, 0.12, 0.04])
        ax_jump.set_facecolor('lightgray')
        self.jump_textbox = TextBox(ax_jump, 'Jump to: ', initial='0')
        self.jump_textbox.on_submit(self.on_jump_click)
        
        # 添加说明文本
        info_text = (
            "Controls:\n"
            "←/h: Previous image\n"
            "→/l: Next image\n"
            "↑/k: Prev original\n"
            "↓/j: Next original\n"
            "PageUp/Down: Jump 100\n"
            "Home/End: First/Last\n"
            "C: Chunk info\n"
            "Q/ESC: Quit"
        )
        self.fig.text(0.02, 0.02, info_text, fontsize=8, 
                     verticalalignment='bottom', 
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def show(self):
        """显示查看器"""
        plt.show()
    
    def __del__(self):
        """析构函数，释放内存"""
        if hasattr(self, 'current_chunk'):
            del self.current_chunk
        if hasattr(self, 'current_chunk_scores'):
            del self.current_chunk_scores
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    parser = argparse.ArgumentParser(description='View preprocessed image data with chunked loading')
    parser.add_argument('--data_folder', type=str, default='maxVit/preprocessed_data',
                        help='Path to the preprocessed data folder')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Chunk size for loading data (default: 100)')
    
    args = parser.parse_args()
    
    # 检查文件夹是否存在
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder does not exist: {args.data_folder}")
        return
    
    # 检查必要的文件是否存在
    required_dirs = ['images_chunks', 'scores_chunks']
    for dir_name in required_dirs:
        dir_path = os.path.join(args.data_folder, dir_name)
        if not os.path.exists(dir_path):
            print(f"Error: Required directory does not exist: {dir_path}")
            return
    
    metadata_path = os.path.join(args.data_folder, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file does not exist: {metadata_path}")
        return
    
    # 创建查看器并显示
    viewer = ChunkedImageViewer(args.data_folder, batch_size=args.batch_size)
    viewer.show()

if __name__ == "__main__":
    main()