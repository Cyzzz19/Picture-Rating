import os
import json
from PIL import Image
import cv2
import numpy as np
import argparse

# 屏幕分辨率
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160

# ========================
# Interactive Image Scorer
# ========================
class InteractiveImageScorer:
    def __init__(self, image_folder, score_file="scores.json"):
        self.image_folder = image_folder
        self.score_file = score_file
        
        # Get all image paths
        self.image_paths = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        print(f"Found {len(self.image_paths)} images")
        
        # Load existing scores
        self.scores = self.load_scores()
        
        # Initialize current image index
        self.current_index = 0
        # Find first unscored image
        self.find_first_unscored()
        
        # Create fullscreen window
        cv2.namedWindow("Image Scorer", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Image Scorer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

    def find_first_unscored(self):
        """Find first unscored image"""
        for i, image_path in enumerate(self.image_paths):
            rel_path = os.path.relpath(image_path, self.image_folder)
            if rel_path not in self.scores:
                self.current_index = i
                return
        # If all are scored, start from first
        self.current_index = 0

    def display_image(self):
        if not self.image_paths:
            print("No images found")
            return

        image_path = self.image_paths[self.current_index]
        rel_path = os.path.relpath(image_path, self.image_folder)
        
        # Display image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot load image: {image_path}")
            return

        # Resize to fit fullscreen (keep aspect ratio)
        h, w = img.shape[:2]
        
        # Calculate scale, use smaller scale to ensure image fits within screen
        scale_w = SCREEN_WIDTH / w
        scale_h = SCREEN_HEIGHT / h
        scale = min(scale_w, scale_h)  # Choose smaller scale
        
        # Resize according to scale
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Create black background
        background = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Calculate center position
        y_offset = (SCREEN_HEIGHT - new_h) // 2
        x_offset = (SCREEN_WIDTH - new_w) // 2
        
        # Place image in center of background
        background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        img_display = background

        # Add current image info
        current_info = f"Image {self.current_index + 1}/{len(self.image_paths)}: {os.path.basename(image_path)}"
        cv2.putText(img_display, current_info, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Add scored count info
        scored_count = 0
        for path in self.image_paths:
            rel_path_check = os.path.relpath(path, self.image_folder)
            if rel_path_check in self.scores:
                scored_count += 1
        
        info = f"Scored: {scored_count}/{len(self.image_paths)}"
        cv2.putText(img_display, info, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Add current score (if scored)
        if rel_path in self.scores:
            current_score = f"Current Score: {self.scores[rel_path]}"
            cv2.putText(img_display, current_score, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # Add operation hints
        hint = "Q:Prev E:Next 0-9:Score Enter:Confirm R:Remove Esc:Exit"
        cv2.putText(img_display, hint, (20, SCREEN_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Show image (fullscreen)
        cv2.imshow("Image Scorer", img_display)
        cv2.waitKey(1)  # Refresh window

    def get_user_input(self):
        print(f"\nCurrent Image: {os.path.basename(self.image_paths[self.current_index])}")
        if os.path.relpath(self.image_paths[self.current_index], self.image_folder) in self.scores:
            current_score = self.scores[os.path.relpath(self.image_paths[self.current_index], self.image_folder)]
            print(f"Current Score: {current_score}")
        
        print("Operations:")
        print("  - Press 0~9 to enter score")
        print("  - Press Enter to confirm score")
        print("  - Press Q for previous image")
        print("  - Press E for next image") 
        print("  - Press R to remove current score")
        print("  - Press Esc to exit")
        print("-" * 50)

        score_str = ""
        while True:
            key = cv2.waitKey(0) & 0xFF
            if ord('0') <= key <= ord('9'):
                digit = chr(key)
                score_str = digit  # Take last key press
                print(f"Score Input: {score_str}")
            elif key == 13:  # Enter
                if score_str:
                    return float(score_str), 'score'
                else:
                    print("Please enter 0~9")
            elif key == ord('q') or key == ord('Q'):
                return None, 'prev'
            elif key == ord('e') or key == ord('E'):
                return None, 'next'
            elif key == ord('r') or key == ord('R'):
                return None, 'remove'
            elif key == 27:  # Esc
                return None, 'quit'

    def run(self):
        while True:
            self.display_image()
            
            user_score, action = self.get_user_input()
            
            if action == 'quit':
                print("Exiting scoring system")
                break
            elif action == 'prev':
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("Already at first image")
            elif action == 'next':
                if self.current_index < len(self.image_paths) - 1:
                    self.current_index += 1
                else:
                    print("Already at last image")
            elif action == 'remove':
                rel_path = os.path.relpath(self.image_paths[self.current_index], self.image_folder)
                if rel_path in self.scores:
                    del self.scores[rel_path]
                    self.save_scores()
                    print(f"Removed score for {rel_path}")
                else:
                    print("Current image not scored, no need to remove")
            elif action == 'score' and user_score is not None:
                # Save score
                rel_path = os.path.relpath(self.image_paths[self.current_index], self.image_folder)
                self.scores[rel_path] = user_score
                self.save_scores()
                print(f"Saved score: {user_score}")
                
                # Auto move to next
                if self.current_index < len(self.image_paths) - 1:
                    self.current_index += 1
                else:
                    print("All images scored!")

        cv2.destroyAllWindows()
        print("Scores saved")

# ========================
# Main function
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="D:/Picture Rating/better", help="Image folder path")
    parser.add_argument("--score_file", type=str, default="D:/Picture Rating/scores.json", help="Score record file")
    args = parser.parse_args()

    scorer = InteractiveImageScorer(
        image_folder=args.image_folder,
        score_file=args.score_file
    )
    scorer.run()