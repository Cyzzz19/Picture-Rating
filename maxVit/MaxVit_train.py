import os
import json
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
import kornia
import kornia.augmentation as K
import gc  # 添加垃圾回收
import numpy as np
import warnings

# 忽略PIL的EXIF警告
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# 设置设备和优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class RandomChunkDataset(Dataset):
    def __init__(self, data_folder, batch_size=256, shuffle_samples=False, chunk_indices=None):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.shuffle_samples = shuffle_samples
        self.images_chunk_dir = os.path.join(data_folder, 'images_chunks')
        self.scores_chunk_dir = os.path.join(data_folder, 'scores_chunks')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载元数据
        metadata_path = os.path.join(data_folder, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_chunks = self.metadata['num_chunks']
        
        # 获取分块文件，限定 chunk_indices（用于训练/验证划分）
        self.chunk_files = []
        self.chunk_batch_info = []
        total_batches = 0
        
        indices = chunk_indices if chunk_indices is not None else range(self.num_chunks)
        for i in indices:
            images_file = os.path.join(self.images_chunk_dir, f'images_chunk_{i:04d}.pt')
            scores_file = os.path.join(self.scores_chunk_dir, f'scores_chunk_{i:04d}.pt')
            if os.path.exists(images_file) and os.path.exists(scores_file):
                try:
                    chunk_size = torch.load(images_file, map_location='cpu', weights_only=True).size(0)
                    batches_in_chunk = (chunk_size + self.batch_size - 1) // self.batch_size
                    
                    self.chunk_files.append({
                        'images_path': images_file,
                        'scores_path': scores_file,
                        'idx': i,
                        'size': chunk_size,
                        'batches': batches_in_chunk
                    })
                    
                    for batch_idx in range(batches_in_chunk):
                        self.chunk_batch_info.append({
                            'chunk_idx': len(self.chunk_files) - 1,  # 映射到 chunk_files 索引
                            'batch_idx': batch_idx
                        })
                    
                    total_batches += batches_in_chunk
                except Exception as e:
                    print(f"Error loading chunk {i}: {e}")
                    continue
            else:
                print(f"Chunk {i} files missing: {images_file} or {scores_file}")
        
        if not self.chunk_files:
            raise ValueError("No valid chunks found")
        
        self.total_batches = total_batches
        self.current_chunk_idx = -1
        self.current_images = None
        self.current_scores = None
        self.current_chunk_size = 0
        self.current_indices = None
        
        print(f"Loaded {self.metadata['total_samples']} total samples from {len(self.chunk_files)} chunks")
        print(f"Total batches: {self.total_batches}, Batch size: {self.batch_size}")

    def __len__(self):
        return self.total_batches

    def __getitem__(self, idx):
        if idx >= self.total_batches:
            raise IndexError(f"Batch index {idx} out of range, total batches: {self.total_batches}")
        
        batch_info = self.chunk_batch_info[idx]
        chunk_idx = batch_info['chunk_idx']
        batch_idx = batch_info['batch_idx']
        
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)
        
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.current_chunk_size)
        
        batch_indices = self.current_indices[start_idx:end_idx]
        batch_images = self.current_images[batch_indices].to(self.device, non_blocking=True)
        batch_scores = self.current_scores[batch_indices].to(self.device, non_blocking=True)
        
        return batch_images, batch_scores

    def _load_chunk(self, chunk_idx):
        """加载指定的数据块并优化内存管理"""
        if self.current_images is not None:
            del self.current_images
            del self.current_scores
            del self.current_indices
            gc.collect()
            torch.cuda.empty_cache()
        
        chunk_info = self.chunk_files[chunk_idx]
        try:
            images_cpu = torch.load(chunk_info['images_path'], map_location='cpu', weights_only=True)
            scores_cpu = torch.load(chunk_info['scores_path'], map_location='cpu', weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load chunk {chunk_idx}: {e}")
        
        self.current_chunk_idx = chunk_idx
        self.current_chunk_size = chunk_info['size']
        self.current_images = images_cpu.float()
        self.current_scores = scores_cpu.float()
        
        if self.shuffle_samples:
            self.current_indices = np.random.permutation(self.current_chunk_size)
        else:
            self.current_indices = np.arange(self.current_chunk_size)
        
        print(f"Loaded chunk {chunk_idx} with {self.current_chunk_size} samples")

# 自定义数据集（用于验证集）
class ImageScoreDataset(Dataset):
    def __init__(self, image_paths, scores, transform=None):
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 修复EXIF问题
            image = Image.open(img_path).convert("RGB")
            image.load()  # 强制加载图像数据
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        score = self.scores[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)

def create_chunk_data_loader(data_folder, batch_size=512, shuffle_samples=False, chunk_indices=None):
    dataset = RandomChunkDataset(
        data_folder, 
        batch_size=batch_size,
        shuffle_samples=shuffle_samples,
        chunk_indices=chunk_indices
    )
    
    def chunk_collate_fn(batch):
        images, scores = batch[0]
        return images, scores  # 数据已在 __getitem__ 中转移到 GPU
    
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=chunk_collate_fn
    )
    
    return data_loader, dataset

# 检查并加载单个文件夹中的数据
def load_single_folder_data(folder_path):
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
def load_all_data(root_folder):
    all_image_paths = []
    all_scores = []
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(root_folder):
        # 检查当前文件夹是否包含scores.json
        if 'scores.json' in files:
            try:
                img_paths, scores = load_single_folder_data(root)
                all_image_paths.extend(img_paths)
                all_scores.extend(scores)
            except Exception as e:
                print(f"Error processing folder {root}: {e}")
                continue
    
    if len(all_image_paths) == 0:
        raise ValueError("No valid data found in any subfolder")
    
    print(f"Total loaded {len(all_image_paths)} images from all subfolders")
    return all_image_paths, all_scores

# 数据增强 - 根据参数决定是否应用增强
def create_transforms(augmentations, is_train=True):
    transform_list = []
    
    if is_train:
        # 基础变换
        transform_list.append(transforms.Resize((224, 224)))
    else:
        # 验证集：基础变换
        transform_list.append(transforms.Resize((224, 224)))
    
    # 转换为张量
    transform_list.append(transforms.ToTensor())
    
    # 标准化
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

# 优化的MaxVitRegressor模型
class MaxVitRegressor(nn.Module):
    def __init__(self):
        super(MaxVitRegressor, self).__init__()
        # 加载预训练 MaxViT-Tiny
        self.backbone = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        
        # 冻结早期层以减少计算量
        for name, param in self.backbone.named_parameters():
            if 'layers.0.' in name or 'layers.1.' in name or 'stem.' in name:  # 冻结stem和前两层
                param.requires_grad = False
        
        # 移除原始分类器，获取特征提取器
        self.backbone.classifier = nn.Identity()  # 完全移除分类器
        
        # 将backbone移动到设备上
        self.backbone = self.backbone.to(device)
        
        # 动态获取特征维度 - 在正确的设备上创建测试输入
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            features = self.backbone(test_input)
            feature_dim = features.shape[1]  # 获取通道数
            print(f"Feature dimension: {feature_dim}")
        
        # 重新定义分类器为回归层
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 减少过拟合
            nn.Linear(512, 1)
        ).to(device)  # 确保回归器也在正确的设备上

    def forward(self, x):
        features = self.backbone(x)
        # 应用回归层
        return self.regressor(features).squeeze(1)

def train_model(data_folder, save_path, epochs=50, lr=1e-3, val_split=0.2):
    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")
    
    # 加载元数据以获取 chunk 总数
    metadata_path = os.path.join(data_folder, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_chunks = metadata['num_chunks']
    
    # 划分训练和验证 chunk
    val_size = int(num_chunks * val_split)
    train_indices = list(range(num_chunks - val_size))
    val_indices = list(range(num_chunks - val_size, num_chunks))
    
    # 创建训练和验证数据加载器
    train_loader, train_dataset = create_chunk_data_loader(
        data_folder, batch_size=512, shuffle_samples=False, chunk_indices=train_indices
    )
    val_loader, val_dataset = create_chunk_data_loader(
        data_folder, batch_size=512, shuffle_samples=False, chunk_indices=val_indices
    )
    
    model = MaxVitRegressor()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    print(f"Training on {device}")
    print(f"Train dataset: {len(train_dataset)} chunks, Val dataset: {len(val_dataset)} chunks")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for chunk_idx, (images, scores_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, scores_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
            
            print(f"  Chunk {chunk_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            if chunk_idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        train_loss /= train_batches if train_batches > 0 else 1
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, scores_batch in val_loader:
                outputs = model(images)
                loss = criterion(outputs, scores_batch)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches if val_batches > 0 else 1
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_path, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            counter = 0
            print(f"New best model saved at {model_path} with val loss: {val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MaxVit model for image scoring using chunked preprocessed data')
    parser.add_argument('--data_folder', type=str, default=r'D:\preprocessed_data', help='Path to preprocessed data folder with chunks')
    parser.add_argument('--save_path', type=str, default=r'maxVit\model', help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    
    try:
        train_model(
            data_folder=args.data_folder,
            save_path=args.save_path,
            epochs=args.epochs,
            lr=args.lr
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()