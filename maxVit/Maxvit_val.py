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
from tqdm import tqdm
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
        w, h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = transforms.Pad(padding, fill=(255, 255, 255))(img)
        return img

# 自定义数据集：不缓存图像，每次即时加载 + 预处理
class ImageScoreDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0.0, dtype=torch.float32)  # dummy label

# 模型定义（与训练一致）
class MaxVitRegressor(nn.Module):
    def __init__(self):
        super(MaxVitRegressor, self).__init__()
        self.backbone = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        for name, param in self.backbone.named_parameters():
            if 'layers.0.' in name or 'layers.1.' in name or 'stem.' in name:
                param.requires_grad = False
        self.backbone.classifier = nn.Identity()
        
        # 动态获取特征维度（在 CPU 上）
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(test_input)
            feature_dim = features.shape[1]
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.to(device)

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features).squeeze(1)

# 验证变换（与训练一致）
val_transform = transforms.Compose([
    ResizeWithPad(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 分批预测函数
def predict(input_folder, output_folder, model_path=r'maxVit\model\best_model.pth', num_samples=50000, batch_size=256, max_batch_size=10000):
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
    
    total_images = len(selected_paths)
    
    # 加载模型（只加载一次）
    model = MaxVitRegressor()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 初始化全局结果列表
    all_results = []
    
    # 创建全局进度条
    pbar = tqdm(total=total_images, desc="Predicting", unit="img")
    
    # 分批处理
    for start in range(0, total_images, max_batch_size):
        end = min(start + max_batch_size, total_images)
        batch_paths = selected_paths[start:end]
        
        pred_dataset = ImageScoreDataset(batch_paths, transform=val_transform)
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        predictions = []
        with torch.no_grad():
            for images, _ in pred_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                batch_preds = outputs.cpu().tolist()
                predictions.extend(batch_preds)
                pbar.update(len(batch_preds))  # 更新进度条
        
        # 配对路径和分数
        batch_results = list(zip(batch_paths, predictions))
        all_results.extend(batch_results)
        
        # 清理内存
        del pred_dataset, pred_loader, predictions
        gc.collect()
        torch.cuda.empty_cache()
    
    pbar.close()
    
    # 全局排序
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    # 保存结果
    os.makedirs(os.path.join(output_folder, 'all'), exist_ok=True)
    for path, score in all_results:
        basename = os.path.basename(path)
        ext = os.path.splitext(basename)[1]
        new_name = f"{score:.4f}_{random.randint(1000, 9999)}{ext}"
        dest_path = os.path.join(output_folder, 'all', new_name)
        try:
            shutil.copy(path, dest_path)
        except Exception as e:
            print(f"Error saving {new_name}: {e}")
    
    print(f"\n✅ Prediction completed. Saved {len(all_results)} images to {output_folder}/all")

if __name__ == "__main__":
    input_folder = r"D:\Images\unrated\better"
    output_folder = r"output/Maxvit_out"
    try:
        predict(input_folder, output_folder)
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()