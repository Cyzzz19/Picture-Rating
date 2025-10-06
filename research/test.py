import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DeepCNNVisualizer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.output_dir = 'cnn_visualization'
        self.feature_maps = {}
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_and_preprocess_image(self):
        """加载并预处理图像"""
        try:
            # 加载图像
            image = Image.open(self.image_path).convert('RGB')
            original_array = np.array(image)
            
            # 保存原始图像
            plt.figure(figsize=(10, 8))
            plt.imshow(original_array)
            plt.title('Original Image')
            plt.axis('off')
            plt.savefig(os.path.join(self.output_dir, '00_original.jpg'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 转换为PyTorch tensor
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
            return image_tensor, original_array
            
        except Exception as e:
            print(f"错误加载图像: {e}")
            return None, None
    
    def create_custom_cnn(self):
        """创建自定义CNN模型用于可视化"""
        class VisualizationCNN(nn.Module):
            def __init__(self):
                super(VisualizationCNN, self).__init__()
                
                # 第一层卷积块
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                
                # 第二层卷积块（深度卷积）
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(128)
                
                # 第三层卷积块（扩张卷积）
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=2)
                self.bn3 = nn.BatchNorm2d(256)
                
                # 第四层卷积块（深度可分离卷积模拟）
                self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, groups=256)
                self.bn4 = nn.BatchNorm2d(512)
                
                # 第五层卷积块（1x1卷积）
                self.conv5 = nn.Conv2d(512, 256, kernel_size=1)
                self.bn5 = nn.BatchNorm2d(256)
                
                # 池化层
                self.pool = nn.MaxPool2d(2, 2)
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                
            def forward(self, x, save_features=True):
                features = {}
                
                # 第一层
                x1 = F.relu(self.bn1(self.conv1(x)))
                features['conv1'] = x1
                x1_pool = self.pool(x1)
                features['pool1'] = x1_pool
                
                # 第二层
                x2 = F.relu(self.bn2(self.conv2(x1_pool)))
                features['conv2'] = x2
                x2_pool = self.pool(x2)
                features['pool2'] = x2_pool
                
                # 第三层（扩张卷积）
                x3 = F.relu(self.bn3(self.conv3(x2_pool)))
                features['conv3'] = x3
                x3_pool = self.pool(x3)
                features['pool3'] = x3_pool
                
                # 第四层（分组卷积）
                x4 = F.relu(self.bn4(self.conv4(x3_pool)))
                features['conv4'] = x4
                x4_pool = self.pool(x4)
                features['pool4'] = x4_pool
                
                # 第五层（1x1卷积）
                x5 = F.relu(self.bn5(self.conv5(x4_pool)))
                features['conv5'] = x5
                x5_pool = self.avg_pool(x5)
                features['global_pool'] = x5_pool
                
                return x5_pool, features
        
        return VisualizationCNN()
    
    def visualize_feature_maps(self, features, layer_name, original_size=(224, 224)):
        """可视化特征图"""
        feature_tensor = features[layer_name].detach().squeeze(0)
        num_channels = feature_tensor.size(0)
        
        # 选择部分通道进行可视化（最多16个）
        num_to_show = min(16, num_channels)
        channels_to_show = torch.linspace(0, num_channels-1, num_to_show).long()
        
        # 创建子图
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, channel_idx in enumerate(channels_to_show):
            if i >= len(axes):
                break
                
            channel_data = feature_tensor[channel_idx].cpu().numpy()
            
            # 归一化
            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
            
            # 上采样到原始尺寸以便观察
            if channel_data.shape != original_size:
                from scipy.ndimage import zoom
                zoom_factors = [original_size[0]/channel_data.shape[0], 
                              original_size[1]/channel_data.shape[1]]
                channel_data = zoom(channel_data, zoom_factors)
            
            axes[i].imshow(channel_data, cmap='viridis')
            axes[i].set_title(f'Channel {channel_idx}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(channels_to_show), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name} - Feature Maps ({num_to_show}/{num_channels} channels)', 
                    fontsize=16, y=0.92)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{layer_name}_feature_maps.jpg'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_convolution_process(self, image_tensor):
        """可视化卷积过程"""
        # 创建一些示例卷积核
        kernels = {
            'edge_detection': torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float(),
            'blur': torch.tensor([[[[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]]]).float(),
            'sharpen': torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]]).float(),
            'gaussian': torch.tensor([[[[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]]]).float()
        }
        
        # 对每个通道应用卷积
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_gray = np.mean(image_np, axis=2)  # 转换为灰度
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 原始图像
        axes[0,0].imshow(image_gray, cmap='gray')
        axes[0,0].set_title('Original (Grayscale)')
        axes[0,0].axis('off')
        
        # 显示卷积核
        for idx, (name, kernel) in enumerate(kernels.items()):
            row = (idx + 1) // 2
            col = (idx + 1) % 2
            axes[row, col*2].imshow(kernel.squeeze(), cmap='coolwarm', vmin=-1, vmax=1)
            axes[row, col*2].set_title(f'{name} Kernel')
            axes[row, col*2].axis('off')
            
            # 应用卷积
            from scipy import ndimage
            convolved = ndimage.convolve(image_gray, kernel.squeeze().numpy(), mode='constant')
            axes[row, col*2+1].imshow(convolved, cmap='viridis')
            axes[row, col*2+1].set_title(f'{name} Result')
            axes[row, col*2+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'convolution_process.jpg'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_pooling_effects(self, feature_maps):
        """可视化池化效果"""
        # 选择第一层的特征图进行池化演示
        if 'conv1' in feature_maps:
            sample_feature = feature_maps['conv1'].detach().squeeze(0)[0].cpu().numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 原始特征图
            axes[0,0].imshow(sample_feature, cmap='viridis')
            axes[0,0].set_title('Original Feature Map')
            axes[0,0].axis('off')
            
            # 最大池化 2x2
            from scipy.ndimage import maximum_filter
            max_pooled = maximum_filter(sample_feature, size=2)[::2, ::2]
            axes[0,1].imshow(max_pooled, cmap='viridis')
            axes[0,1].set_title('Max Pooling 2x2')
            axes[0,1].axis('off')
            
            # 平均池化 2x2
            from scipy.ndimage import uniform_filter
            avg_pooled = uniform_filter(sample_feature, size=2)[::2, ::2]
            axes[0,2].imshow(avg_pooled, cmap='viridis')
            axes[0,2].set_title('Average Pooling 2x2')
            axes[0,2].axis('off')
            
            # 最大池化 4x4
            max_pooled_4 = maximum_filter(sample_feature, size=4)[::4, ::4]
            axes[1,0].imshow(max_pooled_4, cmap='viridis')
            axes[1,0].set_title('Max Pooling 4x4')
            axes[1,0].axis('off')
            
            # 全局平均池化
            global_avg = np.mean(sample_feature) * np.ones_like(sample_feature[::8, ::8])
            axes[1,1].imshow(global_avg, cmap='viridis')
            axes[1,1].set_title('Global Average Pooling')
            axes[1,1].axis('off')
            
            # 特征图尺寸变化对比
            sizes = [sample_feature.shape, max_pooled.shape, max_pooled_4.shape]
            axes[1,2].text(0.1, 0.5, f'尺寸变化:\n原始: {sizes[0]}\n2x2池化: {sizes[1]}\n4x4池化: {sizes[2]}', 
                          fontsize=12, transform=axes[1,2].transAxes)
            axes[1,2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pooling_effects.jpg'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def create_feature_evolution_chart(self, feature_maps):
        """创建特征演化图表"""
        layers = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4', 'conv5']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, layer_name in enumerate(layers):
            if layer_name in feature_maps:
                # 获取特征图的统计信息
                feature = feature_maps[layer_name].detach().squeeze(0)
                channel_means = feature.mean(dim=(1, 2)).cpu().numpy()
                
                axes[i].hist(channel_means, bins=20, alpha=0.7, color='skyblue')
                axes[i].set_title(f'{layer_name}\nChannels: {feature.size(0)}\nSize: {feature.size(1)}x{feature.size(2)}')
                axes[i].set_xlabel('Channel Mean')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_evolution.jpg'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_visualization(self):
        """运行完整的可视化流程"""
        print("开始CNN可视化流程...")
        
        # 加载图像
        image_tensor, original_array = self.load_and_preprocess_image()
        if image_tensor is None:
            return
        
        print("图像加载和预处理完成")
        
        # 创建CNN模型
        model = self.create_custom_cnn()
        model.eval()
        
        print("CNN模型创建完成")
        
        # 可视化卷积过程
        self.visualize_convolution_process(image_tensor)
        print("卷积过程可视化完成")
        
        # 前向传播获取特征图
        with torch.no_grad():
            output, feature_maps = model(image_tensor)
        
        print("前向传播完成，获取特征图")
        
        # 可视化各层特征图
        layers_to_visualize = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4', 'conv5']
        
        for layer_name in layers_to_visualize:
            if layer_name in feature_maps:
                self.visualize_feature_maps(feature_maps, layer_name)
                print(f"可视化完成: {layer_name}")
        
        # 可视化池化效果
        self.visualize_pooling_effects(feature_maps)
        print("池化效果可视化完成")
        
        # 创建特征演化图表
        self.create_feature_evolution_chart(feature_maps)
        print("特征演化图表创建完成")
        
        # 创建总结报告
        self.create_summary_report(feature_maps)
        print("总结报告创建完成")
        
        print(f"\n所有可视化完成！结果保存在: {self.output_dir}")
    
    def create_summary_report(self, feature_maps):
        """创建总结报告"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 特征图尺寸变化
        layers = []
        sizes = []
        channels = []
        
        for layer_name, feature in feature_maps.items():
            if 'pool' in layer_name or 'conv' in layer_name:
                feat = feature.detach().squeeze(0)
                layers.append(layer_name)
                sizes.append(feat.size(1) * feat.size(2))  # 面积
                channels.append(feat.size(0))
        
        # 尺寸变化图
        axes[0,0].plot(layers, sizes, 'o-', linewidth=2, markersize=8)
        axes[0,0].set_title('Feature Map Size Evolution')
        axes[0,0].set_ylabel('Spatial Size (width × height)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 通道数变化图
        axes[0,1].plot(layers, channels, 's-', color='red', linewidth=2, markersize=8)
        axes[0,1].set_title('Channel Dimension Evolution')
        axes[0,1].set_ylabel('Number of Channels')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 计算感受野增长（近似）
        receptive_field = [1]
        for layer_name in layers:
            if 'pool' in layer_name:
                receptive_field.append(receptive_field[-1] * 2)
            else:
                receptive_field.append(receptive_field[-1] + 2)
        
        axes[1,0].plot(['Input'] + layers, receptive_field, '^-', color='green', linewidth=2, markersize=8)
        axes[1,0].set_title('Receptive Field Growth')
        axes[1,0].set_ylabel('Receptive Field Size')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 参数数量估计
        params_text = "CNN架构总结:\n\n"
        params_text += "层类型演变:\n"
        params_text += "Conv3x3 → Pool → Conv3x3 → Pool → \n"
        params_text += "DilatedConv → Pool → GroupConv → Pool → Conv1x1\n\n"
        params_text += "特征抽象过程:\n"
        params_text += "低级特征(边缘,纹理) → 中级特征(形状) → 高级特征(语义)"
        
        axes[1,1].text(0.05, 0.95, params_text, fontsize=12, verticalalignment='top')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cnn_summary_report.jpg'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    print("深度卷积神经网络可视化演示")
    print("=" * 50)
    
    # 获取图像路径
    image_path = input("请输入图像路径（直接回车使用示例图像）: ").strip()
    
    if not image_path or not os.path.exists(image_path):
        # 创建示例图像
        print("创建示例图像...")
        demo_image = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        # 添加一些图案
        demo_image[50:150, 50:150] = [100, 150, 200]  # 彩色方块
        demo_image[180:250, 100:200] = [200, 100, 150]  # 另一个方块
        image_path = 'demo_cnn_image.jpg'
        Image.fromarray(demo_image).save(image_path)
        print(f"已创建示例图像: {image_path}")
    
    # 创建可视化器并运行
    visualizer = DeepCNNVisualizer(image_path)
    visualizer.run_visualization()
    
    print("\n可视化文件说明:")
    print("00_original.jpg - 原始图像")
    print("convolution_process.jpg - 卷积过程演示")
    print("pooling_effects.jpg - 池化效果对比")
    print("conv1-pool5_*.jpg - 各层特征图可视化")
    print("feature_evolution.jpg - 特征演化统计")
    print("cnn_summary_report.jpg - CNN架构总结")

if __name__ == "__main__":
    main()