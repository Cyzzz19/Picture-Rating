import os
import subprocess
import sys

def install_requirements():
    """安装必要的依赖"""
    requirements = [
        'pyinstaller',
        'opencv-python',
        'pillow',
        'numpy'
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ 已安装 {package}")
        except subprocess.CalledProcessError:
            print(f"✗ 安装 {package} 失败")

def build_exe():
    """构建exe文件"""
    script_name = 'Scorer GUI/scorer_gui.py'  # 你的主脚本文件名
    exe_name = 'ImageScorer'
    
    # PyInstaller命令
    cmd = [
        'pyinstaller',
        '--onefile',           # 打包成单个exe
        '--windowed',          # 不显示控制台窗口
        '--name', exe_name,    # exe名称
        '--add-data', '*.spec;.',  # 包含spec文件（如果有）
        '--hidden-import', 'PIL._tkinter_finder',
        '--hidden-import', 'tkinter',
        '--hidden-import', 'cv2',
        '--hidden-import', 'numpy',
        '--clean',             # 清理临时文件
        script_name
    ]
    
    try:
        print("开始打包...")
        subprocess.check_call(cmd)
        print("✓ 打包完成！")
        print(f"exe文件位置: dist/{exe_name}.exe")
    except subprocess.CalledProcessError as e:
        print(f"✗ 打包失败: {e}")

if __name__ == '__main__':
    print("=== Python脚本打包工具 ===\n")
    
    # 检查是否需要安装依赖
    response = input("是否安装依赖包？(y/n): ").lower()
    if response == 'y':
        install_requirements()
    
    # 开始打包
    build_exe()
    
    input("\n按Enter键退出...")