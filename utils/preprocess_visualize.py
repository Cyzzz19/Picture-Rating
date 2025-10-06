import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor
import argparse

# Swin Transformer 模型配置
SWIN_MODEL_NAME = "./swing transformer/model/swin-base-patch4-window7-224"

class SwinImagePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)

    def preprocess_image(self):
        # 加载图像并转换为 RGB
        image = Image.open(self.image_path).convert("RGB")
        
        # 使用 Swin Transformer 的预处理器处理图像（调整为 224x224 并归一化）
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        # 移除 batch 维度以便可视化
        pixel_values = pixel_values.squeeze(0)  # Shape: [C, H, W]
        
        # 将 tensor 转换回 numpy 数组以便显示
        # 反转归一化以便可视化（近似还原到 [0, 1] 范围）
        image_array = pixel_values.permute(1, 2, 0).numpy()  # Shape: [H, W, C]
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        return image, image_array

    def visualize(self):
        # 预处理图像
        original_image, preprocessed_image = self.preprocess_image()
        
        # 创建可视化图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 显示原始图像
        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # 显示预处理后的图像
        ax2.imshow(preprocessed_image)
        ax2.set_title("Swin Preprocessed Image (224x224)")
        ax2.axis("off")
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Swin Transformer image preprocessing")
    parser.add_argument("--image_path", type=str, default="./rated/1B.png", 
                        help="Path to input image")
    args = parser.parse_args()
    
    # 确保图像文件存在
    if not os.path.exists(args.image_path):
        print(f"Image not found at {args.image_path}")
        return
    
    # 创建预处理器实例并可视化
    preprocessor = SwinImagePreprocessor(args.image_path)
    preprocessor.visualize()

if __name__ == "__main__":
    main()