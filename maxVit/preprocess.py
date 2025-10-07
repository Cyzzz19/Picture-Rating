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
from itertools import product
import kornia
import kornia.augmentation as K
from tqdm import tqdm
import warnings

# 忽略PIL的EXIF警告
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# 设置设备和优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

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

def preprocess_and_save_data(root_folder, output_folder, augmentations, batch_size=32, save_batch_size=10000):
    """预处理数据并保存到磁盘，使用GPU加速和分块保存"""
    # 检查根文件夹是否存在
    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")
    
    # 创建保存路径
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载所有数据
    image_paths, scores = load_all_data(root_folder)
    
    if len(image_paths) == 0:
        raise ValueError("No images found or no matching scores in JSON files")
    
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
    print(f"Generated {num_combinations} augmentation combinations")
    
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
    
    print(f"Preprocessing {len(dataset)} images with {num_combinations} augmentations each...")
    print(f"Total expected samples: {len(dataset) * num_combinations}")
    
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
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=len(dataset), desc="Preprocessing Images", unit="images")
    
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
            
            print(f"Saved chunk {chunk_count} - Images: {current_images.size(0)}, Scores: {current_scores.size(0)}")
            
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
        
        print(f"Saved final chunk {chunk_count} - Images: {current_images.size(0)}, Scores: {current_scores.size(0)}")
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
    
    print(f"Preprocessing completed!")
    print(f"Saved {chunk_count} chunks to {images_chunk_dir} and {scores_chunk_dir}")
    print(f"Total samples processed: {total_processed}")

def parse_augmentations(aug_str):
    """解析增强参数字符串"""
    if aug_str is None:
        return ['resize', 'flip', 'rotation', 'jitter']  # 默认增强
    return aug_str.lower().split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for MaxVit training with GPU acceleration')
    parser.add_argument('--data_folder', type=str, default=r"D:\Images\images to train", help='Root folder containing subfolders with images and scores.json')
    parser.add_argument('--output_folder', type=str, default=r'maxVit\preprocessed_data', help='Path to save the preprocessed data')
    parser.add_argument('--augmentations', type=str, default="resize,flip,noise", help='Comma-separated list of augmentations to apply: resize,flip,rotation,jitter,noise')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for preprocessing')
    parser.add_argument('--save_batch_size', type=int, default=128, help='Number of samples to save in each chunk')
    
    args = parser.parse_args()
    
    # 解析增强参数
    augmentations = parse_augmentations(args.augmentations)
    
    try:
        preprocess_and_save_data(
            root_folder=args.data_folder,
            output_folder=args.output_folder,
            augmentations=augmentations,
            batch_size=args.batch_size,
            save_batch_size=args.save_batch_size
        )
    except Exception as e:
        print(f"Preprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()