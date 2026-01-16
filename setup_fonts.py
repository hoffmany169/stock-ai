#!/usr/bin/env python3
import os
import subprocess

def setup_chinese_fonts():
    """设置中文字体"""
    print("Setting up Chinese fonts for Ubuntu...")
    
    # 创建字体缓存目录
    os.makedirs(os.path.expanduser("~/.fonts"), exist_ok=True)
    
    # 安装字体
    fonts_to_install = [
        "fonts-wqy-zenhei",
        "fonts-wqy-microhei",
        "fonts-noto-cjk",
        "ttf-mscorefonts-installer"  # 微软字体
    ]
    
    print("Installing Chinese fonts...")
    for font in fonts_to_install:
        try:
            subprocess.run(["sudo", "apt", "install", "-y", font], check=True)
            print(f"Installed: {font}")
        except:
            print(f"Failed to install: {font}")
    
    # 更新字体缓存
    print("Updating font cache...")
    subprocess.run(["fc-cache", "-fv"])
    
    print("\nFont setup completed!")
    print("You may need to restart the application for changes to take effect.")

if __name__ == "__main__":
    setup_chinese_fonts()