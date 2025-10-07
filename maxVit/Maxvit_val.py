import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import maxvit_t, MaxVit_T_Weights
from PIL import Image
import shutil
import gc

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义 ResizeWithPad 类（与训练代码一致）
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

# 自定义数据集（为预测修改，仅加载图片）
class ImageScoreDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.cached_images = None  # 缓存图像（可选）

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cached_images is not None:
            image = self.cached_images[idx]
        else:
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                image.load()  # 强制加载图像数据
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(0.0, dtype=torch.float32)  # Dummy 分数，仅为兼容

    def cache_images(self):
        """缓存所有图像到内存（如果内存足够）"""
        print("Caching images to memory...")
        self.cached_images = []
        for img_path in self.image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                image.load()
                self.cached_images.append(image)
            except Exception as e:
                print(f"Error caching image {img_path}: {e}")
                self.cached_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        print(f"Cached {len(self.cached_images)} images")

# 模型定义（与训练代码一致）
class MaxVitRegressor(nn.Module):
    def __init__(self):
        super(MaxVitRegressor, self).__init__()
        self.backbone = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        
        # 冻结早期层
        for name, param in self.backbone.named_parameters():
            if 'layers.0.' in name or 'layers.1.' in name or 'stem.' in name:
                param.requires_grad = False
        
        self.backbone.classifier = nn.Identity()
        
        # 动态获取特征维度（在 CPU 上计算）
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)  # 在 CPU 上
            features = self.backbone(test_input)
            feature_dim = features.shape[1]
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        # 将整个模型转移到 device
        self.to(device)

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features).squeeze(1)

# 验证变换（与训练数据块一致）
val_transform = transforms.Compose([
    ResizeWithPad(224),  # 使用 ResizeWithPad 替代 Resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 预测函数
def predict(input_folder, output_folder, model_path=r'maxVit\model\best_model.pth', num_samples=1000, batch_size=32):
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    # 收集所有图像路径
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images in total")
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the input folder")
    
    # 随机选择 num_samples 个图片
    if len(image_paths) < num_samples:
        print(f"Not enough images, using all {len(image_paths)} images")
        selected_paths = image_paths
    else:
        selected_paths = random.sample(image_paths, num_samples)
    
    # 创建数据集和 DataLoader
    pred_dataset = ImageScoreDataset(selected_paths, transform=val_transform)
    
    # 缓存图像（如果内存足够）
    try:
        pred_dataset.cache_images()
    except Exception as e:
        print(f"Failed to cache images: {e}, proceeding without caching")
    
    pred_loader = DataLoader(
        pred_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # 单进程避免开销
        pin_memory=False  # 在 to(device) 中手动处理
    )
    
    # 加载模型
    model = MaxVitRegressor()  # 已转移到 device
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(pred_loader):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions.extend(outputs.cpu().tolist())
            print(f"Processed batch {batch_idx+1}/{len(pred_loader)}")
            gc.collect()
            torch.cuda.empty_cache()
    
    # 配对路径和分数
    results = list(zip(selected_paths, predictions))
    
    # 按分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 保存预测结果（原始图片）
    os.makedirs(os.path.join(output_folder, 'all'), exist_ok=True)
    for i, (path, score) in enumerate(results, 1):
        basename = os.path.basename(path)
        ext = os.path.splitext(basename)[1]
        new_name = f"{score:.4f}{ext}"
        dest_path = os.path.join(output_folder, 'all', new_name)
        try:
            shutil.copy(path, dest_path)
            print(f"Saved {new_name}")
        except Exception as e:
            print(f"Error saving {new_name}: {e}")
    
    print(f"Prediction completed. Saved {len(results)} images to {output_folder}/all")

if __name__ == "__main__":
    input_folder = r"D:\Image grabber\thumbnails"
    output_folder = r"output/Maxvit_out"
    try:
        predict(input_folder, output_folder)
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()