import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel, ViTImageProcessor, SwinModel, AutoImageProcessor
from PIL import Image
import cv2
import numpy as np
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm  # 添加进度条库

# 模型配置
VIT_MODEL_NAME = "./vision transformer/model/vit-base-patch16-384"
SWIN_MODEL_NAME = "./swing transformer/model/swin-base-patch4-window7-224"
# ========================
# 1. 数据集类
# ========================
class ImageScoringDataset(Dataset):
    def __init__(self, image_folder, score_file, processor, device):
        self.image_folder = image_folder
        self.processor = processor
        self.device = device
        
        # 加载评分
        with open(score_file, 'r') as f:
            self.scores = json.load(f)
        
        # 获取所有有评分的图片路径
        self.image_paths = []
        for rel_path in self.scores:
            full_path = os.path.join(self.image_folder, rel_path)
            if os.path.exists(full_path):
                self.image_paths.append((full_path, self.scores[rel_path]))
        
        print(f"Loaded {len(self.image_paths)} scored images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, score = self.image_paths[idx]
        
        # 处理图片
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)  # 移除batch维度
        
        # 获取原始尺寸
        original_image = Image.open(image_path)
        original_size = torch.tensor(list(original_image.size), dtype=torch.float32)  # [w, h]
        
        return pixel_values, torch.tensor(score, dtype=torch.float32), original_size

# ========================
# 2. ViT模型
# ========================
class ViTImageScorer(nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_MODEL_NAME)
        self.hidden_size = self.vit.config.hidden_size
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # 回归头：输入 = 图像特征 + 4个分辨率特征
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size + 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.user_id = 0

    def forward(self, pixel_values, original_size):
        outputs = self.vit(pixel_values)
        image_features = outputs.last_hidden_state
        # 取CLS token对应的特征
        cls_features = image_features[:, 0, :]  # [B, D]

        # 添加分辨率特征
        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680  # 归一化
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([cls_features, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

# ========================
# 3. Swin Transformer模型
# ========================
class SwinImageScorer(nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        self.swin = SwinModel.from_pretrained(SWIN_MODEL_NAME)
        self.hidden_size = self.swin.config.hidden_size
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # 回归头：输入 = 图像特征 + 4个分辨率特征
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size + 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.user_id = 0

    def forward(self, pixel_values, original_size):
        outputs = self.swin(pixel_values)
        # Swin Transformer的输出结构
        image_features = outputs.last_hidden_state
        # 取所有patch的平均作为图像表示
        pooled_features = torch.mean(image_features, dim=1)  # [B, D]

        # 添加分辨率特征
        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680  # 归一化
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([pooled_features, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

# ========================
# 4. 深度卷积神经网络模型
# ========================
class CNNImageScorer(nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        # 简单的CNN架构
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应全局平均池化
        )
        
        self.hidden_size = 256
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # 回归头：输入 = 图像特征 + 4个分辨率特征
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size + 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.user_id = 0

    def forward(self, pixel_values, original_size):
        # pixel_values shape: [B, 3, H, W]
        conv_features = self.conv_layers(pixel_values)  # [B, 256, 1, 1]
        conv_features = conv_features.view(conv_features.size(0), -1)  # [B, 256]

        # 添加分辨率特征
        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680  # 归一化
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([conv_features, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

# ========================
# 5. 训练器类
# ========================
class ModelTrainer:
    def __init__(self, model_type, image_folder, score_file, model_save_path, device):
        self.model_type = model_type
        self.image_folder = image_folder
        self.score_file = score_file
        self.model_save_path = model_save_path
        self.device = device
        
        # 初始化模型
        if model_type == 'vit':
            self.processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
            self.model = ViTImageScorer().to(self.device)
        elif model_type == 'swin':
            self.processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)
            self.model = SwinImageScorer().to(self.device)
        elif model_type == 'cnn':
            # CNN不需要特殊处理器，使用简单的预处理
            self.processor = None
            self.model = CNNImageScorer().to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 优化器使用更大的学习率和权重衰减
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.train_times = []
        
        # 检查是否有预训练模型
        if os.path.exists(self.model_save_path):
            try:
                self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                print(f"Loaded pre-trained model: {self.model_save_path}")
            except Exception as e:
                print(f"Failed to load model, training new: {e}")
        else:
            print("Training new model")
        
        # 创建保存目录
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def prepare_data(self, batch_size=8):
        if self.model_type == 'cnn':
            # 为CNN创建简单的数据集（不使用transformers处理器）
            dataset = self.create_cnn_dataset()
        else:
            dataset = ImageScoringDataset(self.image_folder, self.score_file, self.processor, self.device)
        
        # 分割训练和验证集
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 优化数据加载器，使用更多worker
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True)
        
        return train_loader, val_loader

    def create_cnn_dataset(self):
        # 为CNN创建特殊的数据集
        with open(self.score_file, 'r') as f:
            scores = json.load(f)
        
        image_paths = []
        for rel_path in scores:
            full_path = os.path.join(self.image_folder, rel_path)
            if os.path.exists(full_path):
                image_paths.append((full_path, scores[rel_path]))
        
        return CNNDataset(image_paths, self.device)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f"Training {self.model_type.upper()}")
        for batch_idx, (pixel_values, targets, original_sizes) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 批量处理，而不是逐个处理
            # 获取当前批次中每个样本的原始尺寸
            batch_losses = []
            for i in range(pixel_values.size(0)):
                pv = pixel_values[i:i+1]  # 保持batch维度
                target = targets[i:i+1]
                original_size = tuple(original_sizes[i].cpu().numpy())
                
                pred_score = self.model(pv, original_size)
                loss = self.criterion(pred_score, target)
                batch_losses.append(loss)
            
            total_batch_loss = sum(batch_losses)
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # 更新进度条
            pbar.set_postfix({'Loss': f'{total_batch_loss.item():.4f}'})
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        self.train_losses.append(avg_loss)
        self.train_times.append(epoch_time)
        
        return avg_loss, epoch_time

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validating {self.model_type.upper()}")
            for pixel_values, targets, original_sizes in pbar:
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                batch_losses = []
                for i in range(pixel_values.size(0)):
                    pv = pixel_values[i:i+1]  # 保持batch维度
                    target = targets[i:i+1]
                    original_size = tuple(original_sizes[i].cpu().numpy())
                    
                    pred_score = self.model(pv, original_size)
                    loss = self.criterion(pred_score, target)
                    batch_losses.append(loss.item())
                
                total_batch_loss = sum(batch_losses)
                total_loss += total_batch_loss
                
                # 更新进度条
                pbar.set_postfix({'Loss': f'{total_batch_loss/len(batch_losses):.4f}'})
        
        avg_loss = total_loss / len(val_loader.dataset)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self, epochs=50, batch_size=8):
        print(f"Starting training for {self.model_type} model...")
        
        train_loader, val_loader = self.prepare_data(batch_size)
        
        for epoch in range(epochs):
            train_loss, train_time = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {train_time:.2f}s")
        
        # 保存模型
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def plot_training_stats(self):
        """绘制训练统计图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.set_title(f'{self.model_type.upper()} - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 验证损失
        ax2.plot(self.val_losses, label='Validation Loss', color='red')
        ax2.set_title(f'{self.model_type.upper()} - Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # 训练时间
        ax3.plot(self.train_times, label='Train Time per Epoch', color='green')
        ax3.set_title(f'{self.model_type.upper()} - Training Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (s)')
        ax3.grid(True)
        
        # 训练vs验证损失对比
        ax4.plot(self.train_losses, label='Train Loss', color='blue')
        ax4.plot(self.val_losses, label='Validation Loss', color='red')
        ax4.set_title(f'{self.model_type.upper()} - Train vs Validation Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.model_type}_training_stats.png", dpi=300, bbox_inches='tight')
        plt.show()

# ========================
# 6. CNN数据集类
# ========================
class CNNDataset(Dataset):
    def __init__(self, image_paths, device):
        self.image_paths = image_paths
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, score = self.image_paths[idx]
        
        # 加载并预处理图片
        image = Image.open(image_path).convert("RGB")
        # 调整大小为CNN输入尺寸
        image = image.resize((224, 224))
        # 转换为tensor并归一化
        image_array = np.array(image).astype(np.float32) / 255.0
        # 转换维度: HWC -> CHW
        image_tensor = torch.tensor(image_array).permute(2, 0, 1)
        
        # 获取原始尺寸
        original_image = Image.open(image_path)
        original_size = torch.tensor(list(original_image.size), dtype=torch.float32)  # [w, h]
        
        return image_tensor, torch.tensor(score, dtype=torch.float32), original_size

# ========================
# 7. 多模型训练器
# ========================
class MultiModelTrainer:
    def __init__(self, image_folder, score_file):
        self.image_folder = image_folder
        self.score_file = score_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def train_all_models(self, epochs=50, batch_size=8):
        model_configs = {
            #'vit': './model/vit_scorer.pth',
            'swin': './model/swin_scorer.pth', 
            #'cnn': './model/cnn_scorer.pth'
        }
        
        results = {}
        
        for model_type, model_path in model_configs.items():
            print(f"\n{'='*50}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*50}")
            
            trainer = ModelTrainer(
                model_type=model_type,
                image_folder=self.image_folder,
                score_file=self.score_file,
                model_save_path=model_path,
                device=self.device
            )
            
            trainer.train(epochs=epochs, batch_size=batch_size)
            trainer.plot_training_stats()
            
            results[model_type] = {
                'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else float('inf'),
                'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf'),
                'avg_train_time': np.mean(trainer.train_times) if trainer.train_times else 0
            }
        
        # 打印汇总结果
        print(f"\n{'='*60}")
        print("TRAINING COMPLETION SUMMARY")
        print(f"{'='*60}")
        for model_type, result in results.items():
            print(f"{model_type.upper():>8} - Train Loss: {result['final_train_loss']:.4f}, "
                  f"Val Loss: {result['final_val_loss']:.4f}, "
                  f"Avg Time: {result['avg_train_time']:.2f}s")
        
        # 绘制所有模型对比图
        self.plot_model_comparison(results)
        
    def plot_model_comparison(self, results):
        """绘制所有模型对比图"""
        model_types = list(results.keys())
        train_losses = [results[m]['final_train_loss'] for m in model_types]
        val_losses = [results[m]['final_val_loss'] for m in model_types]
        avg_times = [results[m]['avg_train_time'] for m in model_types]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 训练损失对比
        ax1.bar(model_types, train_losses, color=['blue', 'red', 'green'])
        ax1.set_title('Final Training Loss Comparison')
        ax1.set_ylabel('Loss')
        ax1.grid(True, axis='y')
        
        # 验证损失对比
        ax2.bar(model_types, val_losses, color=['blue', 'red', 'green'])
        ax2.set_title('Final Validation Loss Comparison')
        ax2.set_ylabel('Loss')
        ax2.grid(True, axis='y')
        
        # 平均训练时间对比
        ax3.bar(model_types, avg_times, color=['blue', 'red', 'green'])
        ax3.set_title('Average Training Time per Epoch')
        ax3.set_ylabel('Time (s)')
        ax3.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

# ========================
# 8. 主函数
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./rated", help="Image folder path")
    parser.add_argument("--score_file", type=str, default="scores.json", help="Score record file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=90, help="Batch size")
    args = parser.parse_args()

    # 创建模型保存目录
    os.makedirs("./model", exist_ok=True)

    # 训练所有模型
    multi_trainer = MultiModelTrainer(
        image_folder=args.image_folder,
        score_file=args.score_file
    )
    multi_trainer.train_all_models(epochs=args.epochs, batch_size=args.batch_size)