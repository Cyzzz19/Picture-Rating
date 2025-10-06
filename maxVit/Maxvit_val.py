import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import maxvit_t, MaxVit_T_Weights
import shutil

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集（为预测修改，不需要真实分数）
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
            # 返回一个空白图像作为占位符
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        # 返回 dummy 分数，仅为兼容 DataLoader
        return image, torch.tensor(0.0, dtype=torch.float32)

# 模型定义（与训练代码一致）
class MaxVitRegressor(nn.Module):
    def __init__(self):
        super(MaxVitRegressor, self).__init__()
        # 加载预训练 MaxViT-Tiny
        self.backbone = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 添加全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 动态获取特征维度
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(test_input)
            pooled = self.global_pool(features)
            feature_dim = pooled.shape[1]
        
        self.regressor = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        return self.regressor(features).squeeze(1)

# 验证变换（与训练代码一致）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 预测函数
def predict(input_folder, output_folder, model_path=r'maxVit\model\best_model.pth', num_samples=1000, batch_size=32):
    # 检查文件夹是否存在
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
    
    # 随机选择 1000 个（如果少于 1000，则使用全部）
    if len(image_paths) < num_samples:
        print(f"Not enough images, using all {len(image_paths)} images")
        selected_paths = image_paths
    else:
        selected_paths = random.sample(image_paths, num_samples)
    
    # 创建数据集和 DataLoader
    pred_dataset = ImageScoreDataset(selected_paths, transform=val_transform)
    pred_loader = DataLoader(
        pred_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 加载模型
    model = MaxVitRegressor().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for images, _ in pred_loader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.cpu().tolist())
    
    # 配对路径和分数
    results = list(zip(selected_paths, predictions))
    
    # 按分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 取出前 16（最高分）和后 16（最低分）
    top_16 = results[:16]
    bottom_16 = results[-16:]
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 保存函数：重命名并复制
    def save_with_score(items, prefix):
        os.makedirs(output_folder + "/" + prefix, exist_ok=True)
        for i, (path, score) in enumerate(items, 1):
            basename = os.path.basename(path)
            ext = os.path.splitext(basename)[1]
            new_name = f"{prefix}_{i}_{score:.4f}{ext}"
            dest_path = os.path.join(output_folder + "/" + prefix, new_name)
            shutil.copy(path, dest_path)
            print(f"Saved {new_name} from {path}")
    
    # 保存 top 和 bottom
    save_with_score(top_16, "top")
    save_with_score(bottom_16, "bottom")
    save_with_score(results, "all")
    print("Prediction and saving completed")

if __name__ == "__main__":
    input_folder = r"C:\Users\Cyz19\Downloads\Downloads"  # 替换为实际输入文件夹路径
    output_folder = "output/Maxvit_out"  # 替换为实际输出文件夹路径
    try:
        predict(input_folder, output_folder)
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()