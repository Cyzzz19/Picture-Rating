import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter
import argparse
from transformers import SwinModel, AutoImageProcessor, SwinConfig

# 模型配置
SWIN_MODEL_NAME = "./swing transformer/model/swin-base-patch4-window7-224"

class SwinImageScorer(torch.nn.Module):
    def __init__(self, output_attentions: bool = True):
        super().__init__()
        config = SwinConfig.from_pretrained(SWIN_MODEL_NAME, 
                                          use_mask_token=False, 
                                          output_attentions=output_attentions)
        self.swin = SwinModel.from_pretrained(SWIN_MODEL_NAME, 
                                            config=config)
        self.hidden_size = config.hidden_size  # 1024 for swin-base
        
        # regressor 结构匹配权重文件
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size + 4, 256),  # 1024 + 4 = 1028
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),  # 匹配 regressor.3.weight: torch.Size([1, 256])
            torch.nn.Sigmoid()
        )

    def forward(self, pixel_values: torch.Tensor, original_size: tuple = None):
        outputs = self.swin(pixel_values)
        features = outputs.last_hidden_state
        pooled_features = torch.mean(features, dim=1)
        
        if original_size:  # 用于 Grad-CAM 和评分
            w, h = original_size
            aspect_ratio = w / h if h > 0 else 1.0
            area = w * h
            size_features = torch.tensor([[w, h, area, aspect_ratio]], 
                                       dtype=torch.float32, 
                                       device=pixel_values.device)
            size_features[:, 0] /= 7680.0
            size_features[:, 1] /= 4320.0
            size_features[:, 2] /= (7680.0 * 4320.0)
            
            combined_features = torch.cat([pooled_features, size_features], dim=1)
            score = self.regressor(combined_features) * 10.0
            return outputs.attentions, score.squeeze(1)
        return outputs.attentions

def visualize_swin_attention(model, processor, image_path, save_dir="./visualization_results", alpha=0.7):
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    inputs = processor(images=image, return_tensors="pt", padding="max_length")
    pixel_values = inputs.pixel_values.to(device)
    
    # 计算有效区域（排除填充）
    img_w, img_h = original_size
    scale = min(224 / img_w, 224 / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    pad_left = (224 - new_w) // 2
    pad_top = (224 - new_h) // 2
    valid_region = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)  # (left, top, right, bottom)
    print(f"Valid region: {valid_region}")

    with torch.no_grad():
        attentions = model(pixel_values)

    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    num_layers = len(attentions)
    layers_to_visualize = [-1]  # 仅可视化最后一层
    layers_to_visualize = [l if l >= 0 else num_layers + l for l in layers_to_visualize]

    for layer_idx in layers_to_visualize:
        attention_maps = attentions[layer_idx][0]  # [num_heads, num_windows, num_patches, num_patches]
        num_heads = attention_maps.shape[0]
        config = model.swin.config
        window_size = config.window_size  # 7
        patch_size = config.patch_size  # 4
        img_tensor = pixel_values[0]
        _, img_h, img_w = img_tensor.shape
        num_windows_h = img_h // (window_size * patch_size)
        num_windows_w = img_w // (window_size * patch_size)
        total_windows = num_windows_h * num_windows_w
        num_patches = window_size * window_size  # 49

        print(f"Layer {layer_idx}: attention_maps shape: {attention_maps.shape}")
        print(f"num_windows_h: {num_windows_h}, num_windows_w: {num_windows_w}, total_windows: {total_windows}")

        # 平均所有头的注意力
        selected_attention = attention_maps.mean(dim=0)  # [num_windows, num_patches, num_patches]
        print(f"Averaged attention shape: {selected_attention.shape}")

        attention_map = np.zeros((num_windows_h * window_size, num_windows_w * window_size))
        for i in range(min(total_windows, selected_attention.shape[0])):
            if len(selected_attention.shape) == 3:
                att = selected_attention[i]  # [num_patches, num_patches]
            else:
                att = selected_attention[i].reshape(window_size, window_size)
            row = i // num_windows_w
            col = i % num_windows_w
            attention_map[row*window_size:(row+1)*window_size, col*window_size:(col+1)*window_size] = att.cpu().numpy()

        # 裁剪热力图到有效区域
        valid_h = valid_region[3] - valid_region[1]
        valid_w = valid_region[2] - valid_region[0]
        attention_map = attention_map[:valid_h // patch_size, :valid_w // patch_size]
        attention_resized = cv2.resize(attention_map, original_size, interpolation=cv2.INTER_NEAREST)
        att_min, att_max = attention_resized.min(), attention_resized.max()
        if att_max > att_min:
            attention_resized = (attention_resized - att_min) / (att_max - att_min)

        # 阈值过滤
        threshold = 0.5
        attention_resized = np.where(attention_resized > threshold, attention_resized, 0)

        # 提取图像边缘
        edge_image = np.array(image.convert('L').filter(ImageFilter.FIND_EDGES))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(image)
        axes[1].imshow(edge_image, cmap='gray', alpha=0.2)
        axes[1].imshow(attention_resized, cmap='jet', alpha=alpha)
        axes[1].set_title(f'Layer {layer_idx} | Averaged Attention')
        axes[1].axis('off')
        cbar = plt.colorbar(axes[1].imshow(attention_resized, cmap='jet', alpha=alpha), ax=axes[1])
        cbar.set_alpha(alpha)
        fig.draw_without_rendering()  # 替换 draw_all
        save_path = os.path.join(save_dir, f"{image_name}_layer{layer_idx}_averaged.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")

def visualize_grad_cam(model, processor, image_path, save_dir="./visualization_results", alpha=0.7):
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    inputs = processor(images=image, return_tensors="pt", padding="max_length")
    pixel_values = inputs.pixel_values.to(device)

    # 计算有效区域
    img_w, img_h = original_size
    scale = min(224 / img_w, 224 / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    pad_left = (224 - new_w) // 2
    pad_top = (224 - new_h) // 2
    valid_region = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
    print(f"Valid region: {valid_region}")

    model.eval()
    pixel_values.requires_grad_(True)

    # 获取 patch_size
    patch_size = model.swin.config.patch_size  # 通常为 4

    # 注册钩子捕获特征图
    activations = None
    def hook_fn(module, input, output):
        nonlocal activations
        # 检查 output 类型，确保提取张量
        if isinstance(output, tuple):
            activations = output[0]  # 假设 hidden_states 是元组的第一个元素
        else:
            activations = output

    hook = model.swin.encoder.layers[-1].register_forward_hook(hook_fn)
    try:
        attentions, score = model(pixel_values, original_size)
        # 确保 activations 是张量
        if activations is None:
            raise RuntimeError("Activations not captured by hook")
        print(f"Activations shape: {activations.shape}")
        
        # 计算评分对特征图的梯度
        gradients = torch.autograd.grad(score.sum(), activations, retain_graph=True)[0]
        print(f"Gradients shape: {gradients.shape}")
    finally:
        hook.remove()  # 移除钩子

    # 重塑激活和梯度
    patch_height = pixel_values.shape[2] // patch_size
    patch_width = pixel_values.shape[3] // patch_size
    activations = activations.permute(0, 2, 1).reshape(1, -1, patch_height, patch_width)
    gradients = gradients.permute(0, 2, 1).reshape(1, -1, patch_height, patch_width)

    # 全局平均池化梯度
    pooled_gradients = torch.mean(gradients, dim=[2, 3])  # [batch, channels]

    # 加权特征图
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[:, i]

    # 生成热力图
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    # 裁剪热力图到有效区域
    valid_h = valid_region[3] - valid_region[1]
    valid_w = valid_region[2] - valid_region[0]
    heatmap = heatmap[:valid_h // patch_size, :valid_w // patch_size]
    heatmap = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_NEAREST)
    heatmap /= np.max(heatmap) + 1e-10  # 归一化

    # 阈值过滤
    threshold = 0.5
    heatmap = np.where(heatmap > threshold, heatmap, 0)

    # 提取图像边缘
    edge_image = np.array(image.convert('L').filter(ImageFilter.FIND_EDGES))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].imshow(edge_image, cmap='gray', alpha=0.2)
    axes[1].imshow(heatmap, cmap='jet', alpha=alpha)
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    cbar = plt.colorbar(axes[1].imshow(heatmap, cmap='jet', alpha=alpha), ax=axes[1])
    cbar.set_alpha(alpha)
    fig.draw_without_rendering()  # 替换 draw_all
    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_gradcam.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved Grad-CAM: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Swin Transformer Visualization")
    parser.add_argument("--image_path", type=str, default=r"C:\Users\Cyz19\Desktop\pic\133700170_p0.jpg", help="Path to the image")
    parser.add_argument("--model_path", type=str, default="./model/swin_scorer.pth", help="Path to trained model")
    parser.add_argument("--save_dir", type=str, default="./visualization_results", help="Directory to save results")
    parser.add_argument("--visualize_type", type=str, choices=['attention', 'gradcam', 'both'], default='both')
    parser.add_argument("--alpha", type=float, default=0.4, help="Transparency of heatmap (0.0 to 1.0)")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"Alpha must be between 0.0 and 1.0, got {args.alpha}")
    
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except PermissionError:
        args.save_dir = os.path.join(os.getcwd(), "visualization_results")
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Permission denied, saving to: {args.save_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading Swin model for visualization...")
    try:
        model = SwinImageScorer(output_attentions=True)
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    if args.visualize_type in ['attention', 'both']:
        visualize_swin_attention(model, processor, args.image_path, args.save_dir, alpha=args.alpha)
    
    if args.visualize_type in ['gradcam', 'both']:
        visualize_grad_cam(model, processor, args.image_path, args.save_dir, alpha=args.alpha)
    
    print(f"\nVisualization completed!")
    print(f"Results saved to: {os.path.abspath(args.save_dir)}")

if __name__ == "__main__":
    main()