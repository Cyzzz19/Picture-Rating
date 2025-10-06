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

# 设置设备和优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

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
        
        # 填充图片（使用黑色填充）
        img = transforms.Pad(padding, fill=(0, 0, 0))(img)
        
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
        
        # 填充图片（使用黑色填充）
        img = transforms.Pad(padding, fill=(0, 0, 0))(img)
        
        return img

# 自定义数据集
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
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为占位符
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        score = self.scores[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)

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
        if 'resize' in augmentations:
            transform_list.append(RandomResizeWithPad(224, scale=(0.8, 1.0)))
        else:
            transform_list.append(ResizeWithPad(224))
        
        if 'flip' in augmentations:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if 'rotation' in augmentations:
            transform_list.append(transforms.RandomRotation(15))
        
        if 'jitter' in augmentations:
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    else:
        # Validation set: basic transforms
        transform_list.append(ResizeWithPad(224))
    
    # Convert to tensor before applying GaussianNoise
    transform_list.append(transforms.ToTensor())
    
    # Apply GaussianNoise after ToTensor if specified
    if is_train and 'noise' in augmentations:
        transform_list.append(GaussianNoise(0, 0.1))
    
    # Normalize after all transformations
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

class GaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        # Check if input is a PIL Image and convert to tensor if necessary
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        # Ensure tensor is float32, but do not move to device
        img = img.to(dtype=torch.float32)
        
        # Generate noise with the same shape as the input
        noise = torch.randn_like(img) * self.std + self.mean
        
        # Add noise and clamp to [0, 1]
        return torch.clamp(img + noise, 0, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
     
# 模型定义
class MaxVitRegressor(nn.Module):
    def __init__(self):
        super(MaxVitRegressor, self).__init__()
        # 加载预训练 MaxViT-Tiny
        self.backbone = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        
        # 检查模型结构
        print("Original classifier:", self.backbone.classifier)
        
        # 移除原始分类器，获取特征提取器
        # MaxViT的classifier通常是: [AdaptiveAvgPool2d, Flatten, Linear]
        # 我们需要保留前面的部分，只移除最后的分类层
        self.backbone.classifier = nn.Identity()  # 完全移除分类器
        
        # 添加全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 动态获取特征维度
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(test_input)
            print(f"Backbone output shape: {features.shape}")
            
            # 应用全局池化
            pooled = self.global_pool(features)
            print(f"Pooled shape: {pooled.shape}")
            
            feature_dim = pooled.shape[1]  # 获取通道数
        
        self.regressor = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        # features现在是4D张量 (B, C, H, W)，需要池化到 (B, C, 1, 1)
        features = self.global_pool(features)
        # 展平到 (B, C)
        features = torch.flatten(features, 1)
        # 应用回归层
        return self.regressor(features).squeeze(1)

# 主函数
def main(root_folder, save_path, augmentations, epochs=50, batch_size=32, lr=1e-3):
    # 检查根文件夹是否存在
    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")
    
    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    
    # 加载所有数据
    image_paths, scores = load_all_data(root_folder)
    
    if len(image_paths) == 0:
        raise ValueError("No images found or no matching scores in JSON files")
    
    # 创建训练和验证数据集
    train_size = int(0.8 * len(image_paths))
    
    # 打乱数据
    combined = list(zip(image_paths, scores))
    random.shuffle(combined)
    image_paths_shuffled, scores_shuffled = zip(*combined)
    
    # 分割数据
    train_image_paths = image_paths_shuffled[:train_size]
    train_scores = scores_shuffled[:train_size]
    val_image_paths = image_paths_shuffled[train_size:]
    val_scores = scores_shuffled[train_size:]
    
    # 创建变换
    train_transform = create_transforms(augmentations, is_train=True)
    val_transform = create_transforms(augmentations, is_train=False)
    
    train_dataset = ImageScoreDataset(
        train_image_paths, 
        train_scores, 
        transform=train_transform
    )
    val_dataset = ImageScoreDataset(
        val_image_paths, 
        val_scores, 
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = MaxVitRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    print(f"Training on {device}, dataset size: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Using augmentations: {augmentations}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, scores_batch in train_loader:
            images, scores_batch = images.to(device), scores_batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, scores_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches if train_batches > 0 else 1
        
        # 验证 - 不使用autocast
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, scores_batch in val_loader:
                images, scores_batch = images.to(device), scores_batch.to(device)
                outputs = model(images)  # 验证时不使用autocast
                loss = criterion(outputs, scores_batch)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches if val_batches > 0 else 1
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 早停
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
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

def parse_augmentations(aug_str):
    """解析增强参数字符串"""
    if aug_str is None:
        return ['resize', 'flip', 'rotation', 'jitter']  # 默认增强
    return aug_str.lower().split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MaxVit model for image scoring')
    parser.add_argument('--data_folder', type=str, default=r"Images\images to train", help='Root folder containing subfolders with images and scores.json')
    parser.add_argument('--save_path', type=str, default=r'maxVit\model', help='Path to save the trained model')
    parser.add_argument('--augmentations', type=str, default="resize,flip,rotation,jitter,noise", help='Comma-separated list of augmentations to apply: resize,flip,rotation,jitter,noise')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # 解析增强参数
    augmentations = parse_augmentations(args.augmentations)
    
    try:
        main(
            root_folder=args.data_folder,
            save_path=args.save_path,
            augmentations=augmentations,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()