import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import ViTModel, ViTImageProcessor, SwinModel, AutoImageProcessor
import argparse
from collections import defaultdict

# 模型配置
VIT_MODEL_NAME = "./vision transformer/model/vit-base-patch16-384"
SWIN_MODEL_NAME = "./swing transformer/model/swin-base-patch4-window7-224"

# ViT模型定义
class ViTImageScorer(torch.nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_MODEL_NAME)
        self.hidden_size = self.vit.config.hidden_size
        self.user_embedding = torch.nn.Embedding(num_users, user_embed_dim)
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size + 4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        self.user_id = 0

    def forward(self, pixel_values, original_size):
        outputs = self.vit(pixel_values)
        image_features = outputs.last_hidden_state
        cls_features = image_features[:, 0, :]

        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([cls_features, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

# Swin Transformer模型定义
class SwinImageScorer(torch.nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        self.swin = SwinModel.from_pretrained(SWIN_MODEL_NAME)
        self.hidden_size = self.swin.config.hidden_size
        self.user_embedding = torch.nn.Embedding(num_users, user_embed_dim)
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size + 4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        self.user_id = 0

    def forward(self, pixel_values, original_size):
        outputs = self.swin(pixel_values)
        image_features = outputs.last_hidden_state
        pooled_features = torch.mean(image_features, dim=1)

        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([pooled_features, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

# CNN模型定义
class CNNImageScorer(torch.nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.hidden_size = 256
        self.user_embedding = torch.nn.Embedding(num_users, user_embed_dim)
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size + 4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        self.user_id = 0

    def forward(self, pixel_values, original_size):
        conv_features = self.conv_layers(pixel_values)
        conv_features = conv_features.view(conv_features.size(0), -1)

        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([conv_features, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

def predict_images(image_folder, output_folder, vit_model_path, swin_model_path, cnn_model_path, num_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print("Loading ViT model...")
    vit_model = ViTImageScorer()
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    vit_model = vit_model.to(device)
    vit_model.eval()
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    
    print("Loading Swin model...")
    swin_model = SwinImageScorer()
    swin_model.load_state_dict(torch.load(swin_model_path, map_location=device))
    swin_model = swin_model.to(device)
    swin_model.eval()
    swin_processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)
    
    print("Loading CNN model...")
    cnn_model = CNNImageScorer()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # 获取图片列表
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    all_images = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))
    
    # 随机选择1000张图片
    if len(all_images) > num_samples:
        selected_images = np.random.choice(all_images, num_samples, replace=False)
    else:
        selected_images = all_images
    
    print(f"Selected {len(selected_images)} images for prediction")
    
    # 存储每个模型的预测结果
    vit_predictions = {}
    swin_predictions = {}
    cnn_predictions = {}
    
    # 预测图片
    for i, img_path in enumerate(selected_images):
        print(f"Processing {i+1}/{len(selected_images)}: {os.path.basename(img_path)}")
        
        try:
            # 加载图片
            img = Image.open(img_path).convert("RGB")
            original_size = img.size
            
            # # ViT预测
            # vit_inputs = vit_processor(images=img, return_tensors="pt").pixel_values.to(device)
            # with torch.no_grad():
            #     vit_score = vit_model(vit_inputs, original_size).cpu().item()
            # vit_predictions[img_path] = vit_score
            
            # Swin预测
            swin_inputs = swin_processor(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                swin_score = swin_model(swin_inputs, original_size).cpu().item()
            swin_predictions[img_path] = swin_score
            
            # # CNN预测
            # # 预处理图片为CNN格式
            # cnn_img = img.resize((224, 224))
            # cnn_array = np.array(cnn_img).astype(np.float32) / 255.0
            # cnn_tensor = torch.tensor(cnn_array).permute(2, 0, 1).unsqueeze(0).to(device)
            # with torch.no_grad():
            #     cnn_score = cnn_model(cnn_tensor, original_size).cpu().item()
            # cnn_predictions[img_path] = cnn_score
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # 为每个模型创建子文件夹并保存结果
    model_folders = {
        'vit': vit_predictions,
        'swin': swin_predictions,
        'cnn': cnn_predictions
    }
    
    for model_name, predictions in model_folders.items():
        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
        
        # 按分数排序
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # 保存前16张和后16张
        top_16 = sorted_predictions[:16]
        bottom_16 = sorted_predictions[-16:]
        
        print(f"Saving top 16 images for {model_name}...")
        for i, (img_path, score) in enumerate(top_16):
            img = Image.open(img_path)
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"top_{i+1:02d}_score_{score:.2f}{ext}"
            output_path = os.path.join(model_output_folder, new_filename)
            img.save(output_path)
        
        print(f"Saving bottom 16 images for {model_name}...")
        for i, (img_path, score) in enumerate(bottom_16):
            img = Image.open(img_path)
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"bottom_{i+1:02d}_score_{score:.2f}{ext}"
            output_path = os.path.join(model_output_folder, new_filename)
            img.save(output_path)
    
    print(f"Saved results to {output_folder}")
    print(f"Total images saved: {len(top_16) + len(bottom_16)} per model = {3 * (len(top_16) + len(bottom_16))} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default=r"D:\test", help="Input image folder")
    parser.add_argument("--output_folder", type=str, default="./predicted_images", help="Output folder for predictions")
    parser.add_argument("--vit_model", type=str, default="./model/vit_scorer.pth", help="ViT model path")
    parser.add_argument("--swin_model", type=str, default="./model/swin_scorer.pth", help="Swin model path")
    parser.add_argument("--cnn_model", type=str, default="./model/cnn_scorer.pth", help="CNN model path")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of images to sample")
    args = parser.parse_args()
    
    predict_images(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        vit_model_path=args.vit_model,
        swin_model_path=args.swin_model,
        cnn_model_path=args.cnn_model,
        num_samples=args.num_samples
    )