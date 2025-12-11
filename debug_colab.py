#!/usr/bin/env python3
"""
Debug script for Colab issues
"""

import os
import sys

def debug_colab_setup():
    print("🔍 DeepSculpt Colab Debug Information")
    print("=" * 50)
    
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Check if we're in the right place
    files_to_check = [
        "deepsculpt/main.py",
        "requirements.txt", 
        "requirements-colab.txt",
        "pyproject.toml"
    ]
    
    print("\n📋 File check:")
    for file in files_to_check:
        exists = "✅" if os.path.exists(file) else "❌"
        print(f"  {exists} {file}")
    
    # Check requirements.txt content
    if os.path.exists("requirements.txt"):
        print("\n📄 requirements.txt content (first 10 lines):")
        with open("requirements.txt", "r") as f:
            lines = f.readlines()[:10]
            for i, line in enumerate(lines, 1):
                print(f"  {i:2d}: {line.strip()}")
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Check if we can import basic packages
    print("\n📦 Package import test:")
    packages_to_test = ["torch", "numpy", "matplotlib", "colorama", "rich"]
    
    for package in packages_to_test:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (not installed)")
    
    # Check if deepsculpt can be imported
    print("\n🎨 DeepSculpt import test:")
    try:
        sys.path.insert(0, 'deepsculpt')
        from core.models.model_factory import PyTorchModelFactory
        print("  ✅ DeepSculpt core imports working!")
    except ImportError as e:
        print(f"  ❌ DeepSculpt import failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Recommendations:")
    
    if not os.path.exists("deepsculpt/main.py"):
        print("  1. Make sure you're in the project root directory")
        print("     Run: %cd /content/deepsculpt (if cloned to /content/)")
    
    if not os.path.exists("requirements-colab.txt"):
        print("  2. Use the Colab-optimized requirements:")
        print("     !pip install colorama rich pyyaml h5py imageio plotly tqdm")
    
    print("  3. Try the minimal installation approach from COLAB_SETUP.md")

if __name__ == "__main__":
    debug_colab_setup()