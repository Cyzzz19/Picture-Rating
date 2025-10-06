import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
# 调整PIL的图片大小限制，解决解压炸弹警告
Image.MAX_IMAGE_PIXELS = 1024 * 1024 * 1024  # 设置为1GB像素限制，足够处理大多数大型图片

# 配置参数
BATCH_SIZE = 32
IMAGE_SIZE = 256
# 强制检查CUDA可用性
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    try:
        # 验证CUDA是否真的可用
        torch.zeros(1).cuda()
        print(f"CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA 检测到但不可用: {e}")
        USE_CUDA = False

# 固定处理路径
ROOT_DIR = r"D:\Picture Rating\better2"
# 定义图片文件扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.jfif')

def get_image_files(root_dir):
    """递归获取所有图片文件路径"""
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, file))
    return image_files

def delete_non_image_files(root_dir):
    """删除所有非图片文件"""
    non_image_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                try:
                    os.remove(file_path)
                    non_image_count += 1
                except Exception as e:
                    print(f"删除非图片文件 {file_path} 失败: {e}")
    return non_image_count

def move_files_to_root(root_dir):
    """将所有子文件夹中的文件移动到主文件夹"""
    moved_count = 0
    image_files = [f for f in get_image_files(root_dir) if os.path.dirname(f) != root_dir]
    
    for file_path in tqdm(image_files, desc="移动文件到主文件夹"):
        try:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(root_dir, file_name)
            
            # 处理同名文件
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(file_name)
                dest_path = os.path.join(root_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            shutil.move(file_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"移动文件 {file_path} 失败: {e}")
    
    return moved_count

def delete_empty_subfolders(root_dir):
    """删除所有空的子文件夹"""
    deleted_count = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_path == root_dir:
                continue
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    deleted_count += 1
            except Exception as e:
                print(f"删除文件夹 {dir_path} 失败: {e}")
    return deleted_count

class ImageDataset(Dataset):
    """图片数据集，用于批量处理"""
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            # 打开图片时添加安全检查
            with Image.open(img_path) as img:
                # 检查图片是否过大，如果过大则调整大小
                max_size = (4096, 4096)  # 合理的最大尺寸
                img.thumbnail(max_size)  # 保持比例缩小到大尺寸以内
                
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, img_path
        except Exception as e:
            print(f"无法处理图片 {img_path}: {e}")
            return None, img_path

def collate_fn(batch):
    """自定义数据拼接函数，过滤掉无效图片"""
    valid_batch = []
    invalid_paths = []
    
    for img, path in batch:
        if img is not None:
            valid_batch.append((img, path))
        else:
            invalid_paths.append(path)
    
    # 删除无效图片
    for path in invalid_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"已删除无法处理的图片: {path}")
        except Exception as e:
            print(f"删除无法处理的图片 {path} 失败: {e}")
    
    if not valid_batch:
        return None, None
        
    imgs = torch.stack([item[0] for item in valid_batch])
    paths = [item[1] for item in valid_batch]
    
    return imgs, paths

def compute_image_hashes(file_paths):
    """计算图片的感知哈希值，优化CUDA使用"""
    # 定义图片转换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(file_paths, transform)
    # 当使用CUDA时可以适当增加worker数量
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2 if USE_CUDA else 0,
        collate_fn=collate_fn,
        pin_memory=USE_CUDA  # 当使用CUDA时启用内存锁定
    )
    
    device = torch.device("cuda" if USE_CUDA else "cpu")
    hashes = []
    
    # 预热CUDA
    if USE_CUDA:
        torch.cuda.empty_cache()
        try:
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            torch.nn.functional.interpolate(dummy, size=(8, 8), mode='bilinear', align_corners=False)
        except:
            pass
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算图片哈希值"):
            images, paths = batch
            
            if images is None:
                continue
            
            # 确保数据移动到正确设备
            images = images.to(device, non_blocking=True)
            
            # 哈希计算 - 优化CUDA操作
            small_imgs = torch.nn.functional.interpolate(
                images, 
                size=(8, 8), 
                mode='bilinear', 
                align_corners=False
            )
            gray_imgs = torch.mean(small_imgs, dim=1)
            avg = torch.mean(gray_imgs, dim=(1, 2), keepdim=True)
            hash_bits = (gray_imgs > avg).float()
            
            # 仅在必要时将数据移回CPU
            hash_bits_cpu = hash_bits.cpu()
            for i in range(hash_bits_cpu.shape[0]):
                hash_str = ''.join(str(int(bit)) for bit in hash_bits_cpu[i].flatten().numpy())
                hashes.append((paths[i], hash_str))
    
    # 清理CUDA内存
    if USE_CUDA:
        torch.cuda.empty_cache()
    
    return hashes

def find_duplicates(hashes):
    """根据哈希值找到重复的图片"""
    hash_map = {}
    for file_path, hash_str in hashes:
        if hash_str not in hash_map:
            hash_map[hash_str] = []
        hash_map[hash_str].append(file_path)
    
    duplicates = []
    for hash_str, files in hash_map.items():
        if len(files) > 1:
            duplicates.append(files)
    
    return duplicates

def delete_duplicates(duplicates):
    """删除重复的图片，保留每组中的第一张"""
    deleted_count = 0
    for group in duplicates:
        for file_path in group[1:]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除图片 {file_path} 失败: {e}")
    return deleted_count

def rename_images(root_dir):
    """将所有图片按数字序列重命名，确保序号连续且无冲突"""
    # 获取去重后的所有图片
    image_files = get_image_files(root_dir)
    if not image_files:
        print("没有找到图片文件")
        return 0
    
    # 收集主目录中已存在的数字文件名，避免冲突
    existing_numbers = set()
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if os.path.isfile(file_path) and file.lower().endswith(IMAGE_EXTENSIONS):
            name, ext = os.path.splitext(file)
            if name.isdigit():
                existing_numbers.add(int(name))
    
    # 按数字序列重命名
    current_number = 1
    for file_path in tqdm(image_files, desc="重命名图片"):
        # 找到下一个可用的序号
        while current_number in existing_numbers:
            current_number += 1
        
        dir_name = os.path.dirname(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        new_name = "C" + f"{current_number}{ext}"
        new_path = os.path.join(dir_name, new_name)
        
        # 如果源文件路径和新路径不同，才进行重命名
        if file_path != new_path:
            try:
                # 先尝试直接重命名
                os.rename(file_path, new_path)
            except FileExistsError:
                # 如果仍然冲突，使用临时文件中转
                temp_name = f"temp_{os.urandom(8).hex()}{ext}"
                temp_path = os.path.join(dir_name, temp_name)
                os.rename(file_path, temp_path)
                os.rename(temp_path, new_path)
            except Exception as e:
                print(f"重命名图片 {file_path} 失败: {e}")
                # 跳过这个文件，继续下一个序号
                current_number += 1
                continue
        
        # 将已使用的序号加入集合，并递增序号
        existing_numbers.add(current_number)
        current_number += 1
    
    return len(image_files)

def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"错误: {ROOT_DIR} 不是一个有效的文件夹")
        return
    
    print(f"开始处理文件夹: {ROOT_DIR}")
    print(f"CUDA 加速: {'已启用 - ' + torch.cuda.get_device_name(0) if USE_CUDA else '未启用'}")
    
    # 1. 删除所有非图片文件
    print("删除非图片文件...")
    non_image_count = delete_non_image_files(ROOT_DIR)
    print(f"已删除 {non_image_count} 个非图片文件")
    
    # 2. 移动子文件夹文件到主文件夹
    print("移动子文件夹中的文件到主文件夹...")
    moved_count = move_files_to_root(ROOT_DIR)
    print(f"已移动 {moved_count} 个文件到主文件夹")
    
    # 3. 删除空文件夹
    print("删除空的子文件夹...")
    deleted_folders_count = delete_empty_subfolders(ROOT_DIR)
    print(f"已删除 {deleted_folders_count} 个空文件夹")
    
    # 4. 获取所有图片文件
    print("查找所有图片文件...")
    image_files = get_image_files(ROOT_DIR)
    print(f"找到 {len(image_files)} 个图片文件")
    
    if not image_files:
        print("没有找到图片文件，程序退出")
        return
    
    # 5. 计算图片哈希值（删除无法处理的图片）
    hashes = compute_image_hashes(image_files)
    
    # 6. 查找重复图片
    print("查找重复图片...")
    duplicates = find_duplicates(hashes)
    print(f"找到 {len(duplicates)} 组重复图片")
    
    # 7. 删除重复图片
    if duplicates:
        print("删除重复图片...")
        deleted_count = delete_duplicates(duplicates)
        print(f"已删除 {deleted_count} 个重复图片")
    else:
        print("没有发现重复图片")
    
    # 8. 重命名所有图片
    print("开始重命名图片...")
    renamed_count = rename_images(ROOT_DIR)
    print(f"已重命名 {renamed_count} 个图片")
    
    print("处理完成!")

if __name__ == "__main__":
    main()
    