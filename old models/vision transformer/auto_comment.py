import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import cv2
import numpy as np
import argparse

# 统一使用384模型
DEFAULT_VIT_MODEL = r'.\model\vit-base-patch16-384'

# 屏幕分辨率
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160

# ========================
# 1. Cross-Attention 模块
# ========================
class PersonalizedCrossAttention(nn.Module):
    def __init__(self, hidden_size, user_embed_dim, num_heads=8):
        super().__init__()
        self.user_proj = nn.Linear(user_embed_dim, hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, image_features, user_emb):
        # image_features: [B, L, D], user_emb: [B, E]
        user_query = self.user_proj(user_emb).unsqueeze(1)  # [B, 1, D]
        attn_output, _ = self.cross_attn(user_query, image_features, image_features)
        return attn_output.squeeze(1)  # [B, D]

# ========================
# 2. 主模型 ViT + Cross-Attention + 回归头
# ========================
class ViTImageScorer(nn.Module):
    def __init__(self, vit_model_name=DEFAULT_VIT_MODEL, user_embed_dim=64, num_users=1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.hidden_size = self.vit.config.hidden_size
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.cross_attn = PersonalizedCrossAttention(self.hidden_size, user_embed_dim)
        
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
        user_emb = self.user_embedding(torch.tensor([self.user_id], device=pixel_values.device))
        personalized_feat = self.cross_attn(image_features, user_emb)

        # 添加分辨率特征
        w, h = original_size
        aspect_ratio = w / h
        area = w * h
        size_features = torch.tensor([[w, h, area, aspect_ratio]], dtype=torch.float32, device=pixel_values.device)
        size_features[:, 0] /= 7680  # 归一化
        size_features[:, 1] /= 4320
        size_features[:, 2] /= (7680 * 4320)
        features = torch.cat([personalized_feat, size_features], dim=1)

        score = self.regressor(features) * 10.0
        return score.squeeze(1)

# ========================
# 3. 数据预处理工具
# ========================
class ImageProcessor:
    def __init__(self, model_name=DEFAULT_VIT_MODEL):  # 统一使用384模型
        self.processor = ViTImageProcessor.from_pretrained(model_name)

    def __call__(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values  # [1, 3, 224, 224]

# ========================
# 4. 评分预测器
# ========================
class ImageScorePredictor:
    def __init__(self, image_folder, model_path="./model/vit_scorer.pth", score_file="scores.json", 
                 prediction_interval=100, learning_rate=1e-5):
        self.image_folder = image_folder
        self.model_path = model_path
        self.score_file = score_file
        self.prediction_interval = prediction_interval  # 每隔多少张图片进行一次迭代
        self.learning_rate = learning_rate  # 学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = ViTImageScorer().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 加载模型
        self.load_model()
        
        # 初始化图像处理器
        self.processor = ImageProcessor()
        
        # 加载评分记录
        self.scores = self.load_scores()
        
        # 获取所有图片路径
        self.image_paths = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
        ])
        
        # 过滤出没有评分的图片
        self.unscored_paths = [
            path for path in self.image_paths
            if os.path.relpath(path, self.image_folder) not in self.scores
        ]
        
        print(f"📊 总共找到 {len(self.image_paths)} 张图片")
        print(f"🔍 未评分图片: {len(self.unscored_paths)} 张")
        print(f"💾 已评分图片: {len(self.scores)} 张")
        
        # 创建全屏窗口
        cv2.namedWindow("Image Scorer", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Image Scorer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def load_model(self):
        """加载预训练模型"""
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"✅ 已加载模型: {self.model_path}")
            except Exception as e:
                print(f"⚠️ 模型加载失败: {e}")
                print("❌ 将使用随机初始化的模型进行预测")
        else:
            print(f"⚠️ 找不到模型文件: {self.model_path}")
            print("❌ 将使用随机初始化的模型进行预测")

    def load_scores(self):
        """加载评分记录"""
        if os.path.exists(self.score_file):
            try:
                with open(self.score_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 评分文件加载失败: {e}")
                return {}
        return {}

    def save_scores(self):
        """保存评分记录"""
        with open(self.score_file, 'w') as f:
            json.dump(self.scores, f, indent=2)

    def save_model(self):
        """保存模型"""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"💾 模型已保存至 {self.model_path}")

    def display_image_with_score(self, image_path, predicted_score=None):
        """显示图片并叠加预测分数"""
        # 用 OpenCV 显示图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 无法加载图片: {image_path}")
            return

        # 调整大小以适应全屏显示（保持宽高比，保证图片完整）
        h, w = img.shape[:2]
        
        # 计算缩放比例，使用较小的比例确保图片完整显示在屏幕内
        scale_w = SCREEN_WIDTH / w
        scale_h = SCREEN_HEIGHT / h
        scale = min(scale_w, scale_h)  # 选择较小的缩放比例
        
        # 按比例缩放
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 创建黑色背景
        background = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (SCREEN_HEIGHT - new_h) // 2
        x_offset = (SCREEN_WIDTH - new_w) // 2
        
        # 将图片放置在背景中央
        background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        img_display = background

        # 添加预测分数文本
        if predicted_score is not None:
            text = f"Predicted Score: {predicted_score:.1f}"
            cv2.putText(img_display, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # 添加提示文本
        hint = "Press Y to accept prediction, 0-9 + Enter to rate, B to train all, Q to quit"
        cv2.putText(img_display, hint, (20, SCREEN_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 显示图片（全屏）
        cv2.imshow("Image Scorer", img_display)
        cv2.waitKey(1)  # 刷新窗口

    def get_user_input(self):
        """获取用户输入 - Y接受预测，数字0-9自定义评分，B训练"""
        print("\n" + "="*60)
        print("⌨️  按 Y 接受预测分数，或输入 0-9 自定义评分，按 Enter 确认")
        print("    按 B 训练所有已评分数据，按 Q 退出")
        print("="*60)

        score_str = ""
        while True:
            key = cv2.waitKey(0) & 0xFF
            if ord('0') <= key <= ord('9'):
                digit = chr(key)
                score_str = digit  # 只取最后一次按键（简化）
                print(f"你输入了: {score_str}")
            elif key == ord('y') or key == ord('Y'):
                return "accept"  # 接受预测
            elif key == ord('b') or key == ord('B'):
                return "train_all"  # 训练所有数据
            elif key == 13:  # Enter
                if score_str:
                    return float(score_str)
                else:
                    print("❌ 请输入 0~9 的数字，或按 Y 接受预测，或按 B 训练")
            elif key == ord('q') or key == ord('Q'):
                return None

    def predict_score(self, pixel_values, original_size):
        """预测图片分数"""
        self.model.eval()
        with torch.no_grad():
            score = self.model(pixel_values, original_size)
        return score.item()

    def train_step(self, pixel_values, target_score, original_size):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        pred_score = self.model(pixel_values, original_size)
        loss = self.criterion(pred_score, target_score)
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred_score.item()

    def train_all_data(self):
        """使用所有已评分数据训练模型"""
        if len(self.scores) == 0:
            print("❌ 没有已评分数据用于训练")
            return

        print(f"🔄 开始使用 {len(self.scores)} 条已评分数据训练模型...")
        
        # 仿照原始代码的训练方式，逐个处理每张图片
        total_loss = 0
        processed_count = 0
        
        for rel_path, score in self.scores.items():
            full_path = os.path.join(self.image_folder, rel_path)
            if os.path.exists(full_path):
                try:
                    # 预处理图片
                    pixel_values = self.processor(full_path).to(self.device)
                    
                    # 获取原始尺寸
                    image = Image.open(full_path)
                    original_size = image.size  # (width, height)
                    
                    # 训练一步
                    target_score = torch.tensor([score], dtype=torch.float32).to(self.device)
                    loss, pred_after = self.train_step(pixel_values, target_score, original_size)
                    total_loss += loss
                    processed_count += 1
                    
                    # 显示进度
                    if processed_count % 50 == 0:
                        print(f"   已训练 {processed_count}/{len(self.scores)} 张图片")
                        
                except Exception as e:
                    print(f"⚠️ 处理图片 {full_path} 时出错: {e}")
                    continue
        
        if processed_count > 0:
            avg_loss = total_loss / processed_count
            print(f"✅ 训练完成，平均损失: {avg_loss:.4f}")
            
            # 保存模型
            self.save_model()
        else:
            print("❌ 没有有效的图片数据用于训练")

    def run(self):
        """运行评分预测流程"""
        processed_count = 0
        print(f"🔍 开始处理 {len(self.unscored_paths)} 张未评分图片")

        for idx, image_path in enumerate(self.unscored_paths):
            rel_path = os.path.relpath(image_path, self.image_folder)
            print(f"\n📊 处理第 {idx+1}/{len(self.unscored_paths)} 张: {rel_path}")

            # 预处理图片
            try:
                pixel_values = self.processor(image_path).to(self.device)
                
                # 获取原始尺寸
                image = Image.open(image_path)
                original_size = image.size  # (width, height)
                
            except Exception as e:
                print(f"❌ 预处理失败: {e}")
                continue

            # 预测分数
            predicted_score = self.predict_score(pixel_values, original_size)
            print(f"🔮 模型预测分: {predicted_score:.2f}")

            # 显示图片（全屏）
            self.display_image_with_score(image_path, predicted_score)

            # 获取用户输入
            user_input = self.get_user_input()
            
            # 处理训练请求
            if user_input == "train_all":
                self.train_all_data()
                # 重新显示当前图片
                self.display_image_with_score(image_path, predicted_score)
                user_input = self.get_user_input()
                
            if user_input is None:
                print("👋 用户退出")
                break

            # 确定最终评分
            if user_input == "accept":
                final_score = predicted_score
                print(f"✅ 接受预测分数: {final_score:.2f}")
            else:
                final_score = user_input
                print(f"✅ 使用自定义分数: {final_score}")

            # 保存评分
            self.scores[rel_path] = final_score
            self.save_scores()

            processed_count += 1

            # 每隔指定数量的图片进行一次迭代提示
            if processed_count % self.prediction_interval == 0:
                print(f"🔄 已处理 {processed_count} 张图片，继续...")

        # 最后保存一次
        self.save_scores()
        cv2.destroyAllWindows()
        print(f"🎉 预测完成！共处理 {processed_count} 张图片")

# ========================
# 5. 主函数
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./images", help="图片文件夹路径")
    parser.add_argument("--model_path", type=str, default="./model/vit_scorer.pth", help="模型加载路径")
    parser.add_argument("--score_file", type=str, default="scores.json", help="评分记录文件")
    parser.add_argument("--prediction_interval", type=int, default=100, help="每隔多少张图片显示一次进度")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="训练学习率")
    args = parser.parse_args()

    predictor = ImageScorePredictor(
        image_folder=args.image_folder,
        model_path=args.model_path,
        score_file=args.score_file,
        prediction_interval=args.prediction_interval,
        learning_rate=args.learning_rate
    )
    predictor.run()