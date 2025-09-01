#!/usr/bin/env python3
"""
AMPml GUI 启动脚本
运行现代化的Web界面版本
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'ampml'))

if __name__ == '__main__':
    try:
        # 使用优化版本GUI
        from ampml.amp_gui_optimized import main
        version = "优化版"
        
        print(f"🧬 启动 AMPml Web界面 ({version})...")
        print("📱 浏览器将自动打开，如果没有请访问: http://localhost:8081")
        main()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        sys.exit(1)
