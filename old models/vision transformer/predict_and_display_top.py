# predict_and_display_top.py
import os
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import cv2
import numpy as np
import random
import argparse
from tqdm import tqdm
import shutil

# 导入模型类（需要与训练脚本中的类保持一致）
DEFAULT_VIT_MODEL = r'.\model\vit-base-patch16-384'

class PersonalizedCrossAttention(nn.Module):
    def __init__(self, hidden_size, user_embed_dim, num_heads=8):
        super().__init__()
        self.user_proj = nn.Linear(user_embed_dim, hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, image_features, user_emb):
        user_query = self.user_proj(user_emb).unsqueeze(1)
        attn_output, _ = self.cross_attn(user_query, image_features, image_features)
        return attn_output.squeeze(1)

class ViTImageScorer(nn.Module):
    def __init__(self, vit_model_name=DEFAULT_VIT_MODEL, user_embed_dim=64, num_users=1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.hidden_size = self.vit.config.hidden_size
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.cross_attn = PersonalizedCrossAttention(self.hidden_size, user_embed_dim)
        
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
        user_emb = self.user_embedding(torch.tensor([self.user_id], device=pixel_values.device))
        personalized_feat = self.cross_attn(image_features, user_emb)

        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([personalized_feat, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

class ImageProcessor:
    def __init__(self, model_name=DEFAULT_VIT_MODEL):
        self.processor = ViTImageProcessor.from_pretrained(model_name)

    def __call__(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values

def load_model(model_path, device):
    """加载训练好的模型"""
    model = ViTImageScorer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_score(model, processor, image_path, device):
    """预测单张图片的分数"""
    try:
        pixel_values = processor(image_path).to(device)
        image = Image.open(image_path)
        original_size = image.size
        with torch.no_grad():
            score = model(pixel_values, original_size)
        return score.item()
    except Exception as e:
        print(f"预测图片 {image_path} 时出错: {e}")
        return 0.0

def copy_images_to_folder(images_and_scores, output_folder, copy_all=True):
    """将图片复制到指定文件夹并按分数命名"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"正在将图片复制到文件夹: {output_folder}")
    
    for i, (image_path, score) in enumerate(images_and_scores):
        try:
            # 获取原文件扩展名
            original_ext = os.path.splitext(image_path)[1].lower()
            
            # 生成新的文件名（分数_原文件名）
            original_filename = os.path.basename(image_path)
            original_name = os.path.splitext(original_filename)[0]
            
            # 使用分数作为前缀，保留原文件名和扩展名
            new_filename = f"{score:.2f}_{original_name}{original_ext}"
            new_filepath = os.path.join(output_folder, new_filename)
            
            # 如果文件已存在，添加数字后缀
            counter = 1
            original_new_filepath = new_filepath
            while os.path.exists(new_filepath):
                name_part = f"{score:.2f}_{original_name}_{counter}{original_ext}"
                new_filepath = os.path.join(output_folder, name_part)
                counter += 1
            
            # 复制文件
            shutil.copy2(image_path, new_filepath)
            
            print(f"已复制: {original_filename} -> {os.path.basename(new_filepath)}")
            
        except Exception as e:
            print(f"复制图片 {image_path} 时出错: {e}")
    
    print(f"图片复制完成，共复制 {len(images_and_scores)} 张图片到 {output_folder}")

def create_collage(images_and_scores, output_size=(3840, 2160), grid_size=(4, 4)):
    """将图片拼接成大图"""
    if len(images_and_scores) == 0:
        return None
    
    # 创建黑色背景
    collage = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    
    # 计算每个子图的尺寸
    cell_width = output_size[0] // grid_size[0]
    cell_height = output_size[1] // grid_size[1]
    
    for i, (image_path, score) in enumerate(images_and_scores):
        if i >= grid_size[0] * grid_size[1]:
            break
            
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            # 计算位置
            row = i // grid_size[0]
            col = i % grid_size[0]
            x = col * cell_width
            y = row * cell_height
            
            # 调整图片大小
            img_resized = cv2.resize(img, (cell_width, cell_height))
            
            # 添加分数文本
            text = f"{score:.1f}"
            cv2.putText(img_resized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 放置到拼图中
            collage[y:y+cell_height, x:x+cell_width] = img_resized
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            continue
    
    return collage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model/vit_scorer.pth", help="模型路径")
    parser.add_argument("--image_folder", type=str, default=r'D:\comic_good', help="图片文件夹路径")
    parser.add_argument("--num_images", type=int, default=200, help="随机选择的图片数量")
    parser.add_argument("--top_k", type=int, default=9, help="显示前K个高分图片")
    parser.add_argument("--output_folder", type=str, default="./selected_images", help="输出文件夹路径")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(args.model_path, device)
    processor = ImageProcessor()
    print("模型加载完成")
    
    # 获取所有图片路径
    print("正在扫描图片...")
    image_paths = []
    for root, dirs, files in os.walk(args.image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 随机选择指定数量的图片
    if len(image_paths) > args.num_images:
        selected_paths = random.sample(image_paths, args.num_images)
        print(f"随机选择了 {args.num_images} 张图片")
    else:
        selected_paths = image_paths
        print(f"图片数量不足，使用全部 {len(image_paths)} 张图片")
    
    # 预测所有选中图片的分数
    print("正在预测图片分数...")
    scores = []
    for image_path in tqdm(selected_paths):
        score = predict_score(model, processor, image_path, device)
        scores.append((image_path, score))
    
    # 按分数从高到低排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 选择前K个
    top_scores = scores[:args.top_k]
    
    print(f"\n前 {args.top_k} 个高分图片:")
    for i, (path, score) in enumerate(top_scores):
        print(f"{i+1:2d}. {os.path.basename(path)} - 分数: {score:.2f}")
    
    # 将选中的图片复制到指定文件夹
    copy_images_to_folder(top_scores, args.output_folder)
    
    # 创建拼图
    print("\n正在创建拼图...")
    collage = create_collage(top_scores)
    
    if collage is not None:
        # 显示拼图
        cv2.namedWindow("Top Rated Images", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Top Rated Images", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Top Rated Images", collage)
        
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("创建拼图失败")

if __name__ == "__main__":
    main()