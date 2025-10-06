import os
import json
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import SwinModel, AutoImageProcessor
import argparse

# 屏幕分辨率
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160

# Swin Transformer模型定义
class SwinImageScorer(torch.nn.Module):
    def __init__(self, user_embed_dim=64, num_users=1):
        super().__init__()
        self.swin = SwinModel.from_pretrained("./swing transformer/model/swin-base-patch4-window7-224")
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

# ========================
# Interactive Image Scorer with Swin Transformer
# ========================
class InteractiveImageScorer:
    def __init__(self, image_folder, score_file="scores.json", swin_model_path=None):
        self.image_folder = image_folder
        self.score_file = score_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载Swin模型
        if swin_model_path:
            print("Loading Swin Transformer model...")
            self.swin_model = SwinImageScorer()
            self.swin_model.load_state_dict(torch.load(swin_model_path, map_location=self.device))
            self.swin_model = self.swin_model.to(self.device)
            self.swin_model.eval()
            self.swin_processor = AutoImageProcessor.from_pretrained("./swing transformer/model/swin-base-patch4-window7-224")
        else:
            self.swin_model = None
            self.swin_processor = None
        
        # 获取所有图片路径（包括子文件夹）
        self.image_paths = self.get_all_image_paths()
        
        print(f"Found {len(self.image_paths)} images")
        
        # 加载现有的评分
        self.scores = self.load_scores()
        
        # 使用Swin模型预测未评分的图片
        self.predict_with_swin()
        
        # 初始化当前索引
        self.current_index = 0
        # 找到第一对未评分的图片
        self.find_first_unscored_pair()
        
        # 创建全屏窗口
        cv2.namedWindow("Swin Image Scorer", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Swin Image Scorer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def get_all_image_paths(self):
        """获取文件夹及其子文件夹中的所有图片路径"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
        image_paths = []
        
        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)

    def load_scores(self):
        if os.path.exists(self.score_file):
            try:
                with open(self.score_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load score file: {e}")
                return {}
        return {}

    def save_scores(self):
        with open(self.score_file, 'w') as f:
            json.dump(self.scores, f, indent=2)

    def predict_with_swin(self):
        """使用Swin模型预测未评分的图片"""
        if not self.swin_model:
            print("Swin model not loaded, skipping prediction")
            return
        
        for image_path in self.image_paths:
            rel_path = os.path.relpath(image_path, self.image_folder)
            if rel_path not in self.scores:
                try:
                    # 加载图片
                    img = Image.open(image_path).convert("RGB")
                    original_size = img.size
                    
                    # 预处理并预测
                    inputs = self.swin_processor(images=img, return_tensors="pt").pixel_values.to(self.device)
                    with torch.no_grad():
                        score = self.swin_model(inputs, original_size).cpu().item()
                    
                    # 保存预测分数
                    self.scores[rel_path] = score
                    print(f"Predicted score for {rel_path}: {score:.2f}")
                except Exception as e:
                    print(f"Error predicting {image_path}: {e}")
        
        # 保存预测结果
        self.save_scores()

    def find_first_unscored_pair(self):
        """找到第一对未评分的图片"""
        # 简单起见，从第一张开始
        self.current_index = 0

    def display_pair(self):
        """显示一对图片及其预测分数"""
        if len(self.image_paths) < 2:
            print("Need at least 2 images to compare")
            return

        # 获取当前两张图片
        idx1 = self.current_index
        idx2 = (self.current_index + 1) % len(self.image_paths)
        
        img_path1 = self.image_paths[idx1]
        img_path2 = self.image_paths[idx2]
        
        rel_path1 = os.path.relpath(img_path1, self.image_folder)
        rel_path2 = os.path.relpath(img_path2, self.image_folder)
        
        # 加载图片
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            print("Cannot load images")
            return

        # 调整图片大小以适应屏幕
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算缩放比例
        max_width = SCREEN_WIDTH // 2 - 50
        max_height = SCREEN_HEIGHT - 200
        
        scale1 = min(max_width / w1, max_height / h1)
        scale2 = min(max_width / w2, max_height / h2)
        
        new_w1 = int(w1 * scale1)
        new_h1 = int(h1 * scale1)
        new_w2 = int(w2 * scale2)
        new_h2 = int(h2 * scale2)
        
        img1_resized = cv2.resize(img1, (new_w1, new_h1))
        img2_resized = cv2.resize(img2, (new_w2, new_h2))
        
        # 创建背景
        background = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # 计算位置
        y_offset1 = (SCREEN_HEIGHT - new_h1) // 2
        x_offset1 = 50
        y_offset2 = (SCREEN_HEIGHT - new_h2) // 2
        x_offset2 = SCREEN_WIDTH // 2 + 50
        
        # 放置图片
        background[y_offset1:y_offset1+new_h1, x_offset1:x_offset1+new_w1] = img1_resized
        background[y_offset2:y_offset2+new_h2, x_offset2:x_offset2+new_w2] = img2_resized
        
        # 添加标签和分数
        score1 = self.scores.get(rel_path1, 0)
        score2 = self.scores.get(rel_path2, 0)
        
        # 图片1信息
        name1 = os.path.basename(img_path1)
        cv2.putText(background, f"1: {name1}", (x_offset1, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(background, f"Score: {score1:.2f}", (x_offset1, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 图片2信息
        name2 = os.path.basename(img_path2)
        cv2.putText(background, f"2: {name2}", (x_offset2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(background, f"Score: {score2:.2f}", (x_offset2, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 添加操作提示
        hint1 = "Y: Accept scores  N: Swap scores  Q: Prev pair  E: Next pair  W: Save & Exit"
        hint2 = "Arrow keys: Adjust scores  0-9: Direct score  Enter: Confirm  Esc: Exit"
        cv2.putText(background, hint1, (50, SCREEN_HEIGHT - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(background, hint2, (50, SCREEN_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示图片
        cv2.imshow("Swin Image Scorer", background)
        cv2.waitKey(1)

    def get_user_input(self):
        print(f"\nComparing images:")
        print(f"Image 1: {os.path.basename(self.image_paths[self.current_index])} (Score: {self.scores.get(os.path.relpath(self.image_paths[self.current_index], self.image_folder), 0):.2f})")
        print(f"Image 2: {os.path.basename(self.image_paths[(self.current_index + 1) % len(self.image_paths)])} (Score: {self.scores.get(os.path.relpath(self.image_paths[(self.current_index + 1) % len(self.image_paths)], self.image_folder), 0):.2f})")
        print("Operations:")
        print("  - Y: Accept current scores")
        print("  - N: Swap scores between images")
        print("  - Q: Previous pair")
        print("  - E: Next pair")
        print("  - W: Write scores and exit")
        print("  - Arrow Up/Down: Adjust first image score")
        print("  - Arrow Right/Left: Adjust second image score")
        print("  - 0~9: Directly set score for selected image")
        print("  - Enter: Confirm selection")
        print("  - Esc: Exit without saving")
        print("-" * 70)

        selected_image = 1  # 1 for first image, 2 for second
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y') or key == ord('Y'):
                return 'accept'
            elif key == ord('n') or key == ord('N'):
                return 'swap'
            elif key == ord('q') or key == ord('Q'):
                return 'prev'
            elif key == ord('e') or key == ord('E'):
                return 'next'
            elif key == ord('w') or key == ord('W'):
                return 'write'
            elif key == 27:  # Esc
                return 'quit'
            elif key == 82:  # Up arrow - increase first image score
                rel_path1 = os.path.relpath(self.image_paths[self.current_index], self.image_folder)
                self.scores[rel_path1] = min(10.0, self.scores.get(rel_path1, 0) + 0.5)
                self.save_scores()
                print(f"Adjusted score for image 1: {self.scores[rel_path1]:.2f}")
                self.display_pair()
            elif key == 84:  # Down arrow - decrease first image score
                rel_path1 = os.path.relpath(self.image_paths[self.current_index], self.image_folder)
                self.scores[rel_path1] = max(0.0, self.scores.get(rel_path1, 0) - 0.5)
                self.save_scores()
                print(f"Adjusted score for image 1: {self.scores[rel_path1]:.2f}")
                self.display_pair()
            elif key == 83:  # Right arrow - increase second image score
                next_idx = (self.current_index + 1) % len(self.image_paths)
                rel_path2 = os.path.relpath(self.image_paths[next_idx], self.image_folder)
                self.scores[rel_path2] = min(10.0, self.scores.get(rel_path2, 0) + 0.5)
                self.save_scores()
                print(f"Adjusted score for image 2: {self.scores[rel_path2]:.2f}")
                self.display_pair()
            elif key == 81:  # Left arrow - decrease second image score
                next_idx = (self.current_index + 1) % len(self.image_paths)
                rel_path2 = os.path.relpath(self.image_paths[next_idx], self.image_folder)
                self.scores[rel_path2] = max(0.0, self.scores.get(rel_path2, 0) - 0.5)
                self.save_scores()
                print(f"Adjusted score for image 2: {self.scores[rel_path2]:.2f}")
                self.display_pair()
            elif ord('0') <= key <= ord('9'):
                digit = chr(key)
                score = float(digit)
                if selected_image == 1:
                    rel_path1 = os.path.relpath(self.image_paths[self.current_index], self.image_folder)
                    self.scores[rel_path1] = score
                else:
                    next_idx = (self.current_index + 1) % len(self.image_paths)
                    rel_path2 = os.path.relpath(self.image_paths[next_idx], self.image_folder)
                    self.scores[rel_path2] = score
                self.save_scores()
                print(f"Set score to {score}")
                self.display_pair()
            elif key == 13:  # Enter - toggle selected image
                selected_image = 2 if selected_image == 1 else 1
                print(f"Selected image: {selected_image}")

    def run(self):
        while True:
            self.display_pair()
            
            action = self.get_user_input()
            
            if action == 'quit':
                print("Exiting without saving")
                break
            elif action == 'write':
                print("Saving scores and exiting")
                self.save_scores()
                break
            elif action == 'prev':
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    self.current_index = len(self.image_paths) - 1
            elif action == 'next':
                self.current_index = (self.current_index + 1) % len(self.image_paths)
            elif action == 'accept':
                # 保持当前分数不变，移动到下一组
                self.current_index = (self.current_index + 1) % len(self.image_paths)
            elif action == 'swap':
                # 交换两个图片的分数
                rel_path1 = os.path.relpath(self.image_paths[self.current_index], self.image_folder)
                next_idx = (self.current_index + 1) % len(self.image_paths)
                rel_path2 = os.path.relpath(self.image_paths[next_idx], self.image_folder)
                
                score1 = self.scores.get(rel_path1, 0)
                score2 = self.scores.get(rel_path2, 0)
                
                self.scores[rel_path1] = score2
                self.scores[rel_path2] = score1
                
                self.save_scores()
                print(f"Swapped scores: {score1:.2f} <-> {score2:.2f}")
                
                # 移动到下一组
                self.current_index = (self.current_index + 1) % len(self.image_paths)

        cv2.destroyAllWindows()
        print("Scores saved")

# ========================
# Main function
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="D:/Picture Rating/rating", help="Image folder path")
    parser.add_argument("--score_file", type=str, default="D:/Picture Rating/scores.json", help="Score record file")
    parser.add_argument("--swin_model", type=str, default="./model/swin_scorer.pth", help="Path to Swin Transformer model")
    args = parser.parse_args()

    scorer = InteractiveImageScorer(
        image_folder=args.image_folder,
        score_file=args.score_file,
        swin_model_path=args.swin_model
    )
    scorer.run()