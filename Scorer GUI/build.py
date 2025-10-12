import os
import subprocess
import sys
from pathlib import Path

def create_spec_file():
    """创建包含递归限制设置的spec文件"""
    spec_content = """# 增加递归限制
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

a = Analysis(
    ['Scorer GUI/scorer_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=['PIL._tkinter_finder', 'tkinter', 'cv2', 'numpy', 'matplotlib.backends.backend_tkagg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ImageScorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    # 写入spec文件
    with open('image_scorer.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    return 'image_scorer.spec'

def build_exe():
    """构建exe文件"""
    exe_name = 'ImageScorer'
    
    try:
        # 先创建spec文件
        spec_file = create_spec_file()
        print(f"已创建spec文件: {spec_file}")
        
        # 使用spec文件进行打包
        cmd = [
            'pyinstaller',
            '--clean',             # 清理临时文件
            spec_file              # 使用生成的spec文件
        ]
        
        print("开始打包...")
        subprocess.check_call(cmd)
        print("✓ 打包完成！")
        print(f"exe文件位置: dist/{exe_name}.exe")
        
        # 可选：删除生成的spec文件（如果不需要保留）
        # os.remove(spec_file)
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 打包失败: {e}")
    except Exception as e:
        print(f"✗ 发生错误: {e}")

if __name__ == '__main__':
    print("=== Python脚本打包工具 ===\n")
    
    # 开始打包
    build_exe()
    
    input("\n按Enter键退出...")
    