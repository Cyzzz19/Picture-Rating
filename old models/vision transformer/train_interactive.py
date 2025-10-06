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
# 4. 交互式训练器
# ========================
class InteractiveTrainer:
    def __init__(self, image_folder, model_save_path="./model/vit_scorer.pth", score_file="scores.json", save_every=10, replay_interval=50, replay_epochs=3):
        self.image_folder = image_folder
        self.model_save_path = model_save_path
        self.score_file = score_file
        self.save_every = save_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_interval = replay_interval  # 每隔多少张图片进行重复训练
        self.replay_epochs = replay_epochs      # 重复训练的轮数
        self.processed_count = 0                # 已处理图片计数器
        
        # 用于存储已处理图片的数据缓存
        self.replay_buffer = []
        # 初始化模型
        self.model = ViTImageScorer().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()

        # 添加标志变量
        self.model_loaded = False

        if os.path.exists(self.model_save_path):
            try:
                self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                print(f"✅ 已加载预训练模型: {self.model_save_path}")
                self.model_loaded = True
            except Exception as e:
                print(f"⚠️ 模型加载失败，使用新模型: {e}")
        else:
            print("🆕 未找到预训练模型，使用初始化权重")

        # 校验模型与评分是否同步
        self.validate_state_consistency()

        # 创建全屏窗口
        cv2.namedWindow("Image Scorer", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Image Scorer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def validate_state_consistency(self):
        """检查模型和评分记录是否同步，如不同步则发出警告"""
        # ✅ 先加载评分
        self.scores = self.load_scores()

        # 然后再做校验
        if not self.model_loaded:
            if len(self.scores) > 0:
                print(f"⚠️ 警告：发现 {len(self.scores)} 条历史评分，但未加载模型！")
                print("💡 建议：")
                print("  1. 检查模型路径是否正确")
                print("  2. 或运行 '预热训练' 用历史数据初始化新模型")

        # 获取所有图片路径
        self.image_paths = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 初始化图像处理器
        self.processor = ImageProcessor()

        # 创建模型保存目录
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def load_scores(self):
        if os.path.exists(self.score_file):
            try:
                with open(self.score_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 评分文件加载失败: {e}")
                return {}
        return {}

    def save_scores(self):
        with open(self.score_file, 'w') as f:
            json.dump(self.scores, f, indent=2)

    def display_image_with_score(self, image_path, predicted_score=None):
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
        hint = "Press 0-9 to rate, Enter to confirm, Q to quit"
        cv2.putText(img_display, hint, (20, SCREEN_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 显示图片（全屏）
        cv2.imshow("Image Scorer", img_display)
        cv2.waitKey(1)  # 刷新窗口

    def get_user_input(self):
        print("\n" + "="*50)
        print("⌨️  请输入评分 (0~9)，按 Enter 确认，按 Q 退出")
        print("="*50)

        score_str = ""
        while True:
            key = cv2.waitKey(0) & 0xFF
            if ord('0') <= key <= ord('9'):
                digit = chr(key)
                score_str = digit  # 只取最后一次按键（简化）
                print(f"你输入了: {score_str}")
            elif key == 13:  # Enter
                if score_str:
                    return float(score_str)
                else:
                    print("❌ 请输入 0~9 的数字")
            elif key == ord('q') or key == ord('Q'):
                return None

    def train_step(self, pixel_values, target_score, original_size):
        self.model.train()
        self.optimizer.zero_grad()
        pred_score = self.model(pixel_values, original_size)
        loss = self.criterion(pred_score, target_score)
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred_score.item()

    def predict_score(self, pixel_values, original_size):
        self.model.eval()
        with torch.no_grad():
            score = self.model(pixel_values, original_size)
        return score.item()

    def run(self):
        trained_count = 0
        print(f"📂 共找到 {len(self.image_paths)} 张图片")

        for idx, image_path in enumerate(self.image_paths):
            rel_path = os.path.relpath(image_path, self.image_folder)
            print(f"\n📊 处理第 {idx+1}/{len(self.image_paths)} 张: {rel_path}")

            # 如果已有评分，跳过
            if rel_path in self.scores:
                print(f"✅ 已有评分: {self.scores[rel_path]}，跳过")
                continue

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
            user_score = self.get_user_input()
            if user_score is None:
                print("👋 用户退出")
                break

            # 实时训练
            target_score = torch.tensor([user_score], dtype=torch.float32).to(self.device)
            loss, pred_after = self.train_step(pixel_values, target_score, original_size)  # 修复：添加original_size参数
            print(f"📉 训练损失: {loss:.4f}, 训练后预测: {pred_after:.2f}")

            # 保存评分
            self.scores[rel_path] = user_score
            self.save_scores()

            trained_count += 1

            # 定期保存模型
            if trained_count % self.save_every == 0:
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"💾 模型已保存至 {self.model_save_path}")

        # 最后保存一次
        torch.save(self.model.state_dict(), self.model_save_path)
        cv2.destroyAllWindows()
        print("🎉 训练完成！")
   
    def add_to_replay_buffer(self, pixel_values, score, original_size):
        """将图片数据添加到重放缓冲区"""
        # 限制缓冲区大小，避免内存过大
        if len(self.replay_buffer) > 1000:
            self.replay_buffer.pop(0)
        
        self.replay_buffer.append({
            'pixel_values': pixel_values.clone().detach(),
            'score': score,
            'original_size': original_size
        })

    def replay_training(self):
        """使用缓存数据进行重复训练"""
        if len(self.replay_buffer) == 0:
            return
        
        print(f"🔄 开始重复训练 {self.replay_epochs} 轮，使用 {len(self.replay_buffer)} 张历史图片")
        
        for epoch in range(self.replay_epochs):
            total_loss = 0
            for item in self.replay_buffer:
                pixel_values = item['pixel_values'].to(self.device)
                target_score = torch.tensor([item['score']], dtype=torch.float32).to(self.device)
                original_size = item['original_size']
                
                loss, _ = self.train_step(pixel_values, target_score, original_size)
                total_loss += loss
            
            avg_loss = total_loss / len(self.replay_buffer)
            print(f"   重复训练轮 {epoch+1}/{self.replay_epochs}, 平均损失: {avg_loss:.4f}")

    # 修改 run 方法中的相关部分
    def run(self):
        trained_count = 0
        print(f"📂 共找到 {len(self.image_paths)} 张图片")

        for idx, image_path in enumerate(self.image_paths):
            rel_path = os.path.relpath(image_path, self.image_folder)
            print(f"\n📊 处理第 {idx+1}/{len(self.image_paths)} 张: {rel_path}")

            # 如果已有评分，跳过
            if rel_path in self.scores:
                print(f"✅ 已有评分: {self.scores[rel_path]}，跳过")
                continue

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

            # 显示图片
            self.display_image_with_score(image_path, predicted_score)

            # 获取用户输入
            user_score = self.get_user_input()
            if user_score is None:
                print("👋 用户退出")
                break

            # 实时训练
            target_score = torch.tensor([user_score], dtype=torch.float32).to(self.device)
            loss, pred_after = self.train_step(pixel_values, target_score, original_size)
            print(f"📉 训练损失: {loss:.4f}, 训练后预测: {pred_after:.2f}")

            # 保存评分和添加到重放缓冲区
            self.scores[rel_path] = user_score
            self.add_to_replay_buffer(pixel_values, user_score, original_size)
            self.save_scores()

            trained_count += 1
            self.processed_count += 1

            # 每隔指定间隔进行重复训练
            if self.processed_count % self.replay_interval == 0 and len(self.replay_buffer) > 0:
                self.replay_training()

            # 定期保存模型
            if trained_count % self.save_every == 0:
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"💾 模型已保存至 {self.model_save_path}")

        # 最后保存一次
        torch.save(self.model.state_dict(), self.model_save_path)
        cv2.destroyAllWindows()
        print("🎉 训练完成！")
# ========================
# 5. 主函数
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./images", help="图片文件夹路径")
    parser.add_argument("--model_path", type=str, default="./model/vit_scorer.pth", help="模型保存路径")
    parser.add_argument("--score_file", type=str, default="scores.json", help="评分记录文件")
    parser.add_argument("--save_every", type=int, default=5, help="每训练多少张保存一次模型")
    parser.add_argument("--replay_interval", type=int, default=50, help="每隔多少张图片进行重复训练")
    parser.add_argument("--replay_epochs", type=int, default=3, help="重复训练的轮数")
    args = parser.parse_args()

    trainer = InteractiveTrainer(
        image_folder=args.image_folder,
        model_save_path=args.model_path,
        score_file=args.score_file,
        save_every=args.save_every,
        replay_interval=args.replay_interval,
        replay_epochs=args.replay_epochs
    )
    trainer.run()