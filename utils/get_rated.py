import json
import shutil
from pathlib import Path

def safe_classify_files(source_dir, json_path, present_dir):
    """
    安全版本的文件分类
    先复制文件到目标文件夹，验证成功后删除原文件
    """
    
    # 创建目标目录
    Path(present_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_keys = set(json.load(f).keys())
    
    print(f"JSON中共有 {len(json_keys)} 个文件名")
    
    # 统计计数器
    moved_count = 0
    ignored_count = 0
    error_count = 0
    
    # 遍历所有文件
    source_path = Path(source_dir)
    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            if file_path.name in json_keys:
                try:
                    destination = Path(present_dir) / file_path.name
                    
                    # 先复制文件
                    shutil.copy2(file_path, destination)
                    
                    # 验证复制是否成功（检查文件大小）
                    if destination.stat().st_size == file_path.stat().st_size:
                        # 复制成功，删除原文件
                        file_path.unlink()
                        print(f"✓ 已移动: {file_path.name}")
                        moved_count += 1
                    else:
                        # 复制失败，删除不完整的副本
                        destination.unlink()
                        print(f"✗ 复制验证失败: {file_path.name}")
                        error_count += 1
                        
                except Exception as e:
                    print(f"✗ 处理文件失败: {file_path.name} - 错误: {e}")
                    error_count += 1
            else:
                print(f"○ 已忽略: {file_path.name}")
                ignored_count += 1
    
    print(f"\n处理完成！")
    print(f"已移动文件: {moved_count} 个")
    print(f"已忽略文件: {ignored_count} 个")
    print(f"处理失败: {error_count} 个")
    print(f"目标文件夹: {present_dir}")

# 使用示例
if __name__ == "__main__":
    source_dir = "./images"      # 源文件夹
    json_path = "scores.json"    # JSON文件
    present_dir = "./rated"        # 存在JSON键的文件目录
    
    safe_classify_files(source_dir, json_path, present_dir)