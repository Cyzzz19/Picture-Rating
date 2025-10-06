import os
import re

def fix_files_simple(folder_path):
    """简化版本，直接处理指定文件夹"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            # 使用正则表达式匹配 数字.扩展名B 的模式
            match = re.match(r'^(\d+)\.(\w+)B$', filename, re.IGNORECASE)
            
            if match:
                number = match.group(1)
                extension = match.group(2)
                new_name = f"{number}B.{extension}"
                new_path = os.path.join(folder_path, new_name)
                
                os.rename(file_path, new_path)
                print(f"重命名: {filename} → {new_name}")

# 使用方法：直接修改下面的路径
folder_path = r"D:\Picture Rating\better"  # 替换为您的实际路径
fix_files_simple(folder_path)