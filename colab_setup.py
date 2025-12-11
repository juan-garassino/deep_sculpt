#!/usr/bin/env python3
"""
Google Colab Setup Script for DeepSculpt
This script installs all required dependencies and sets up the environment for DeepSculpt in Google Colab.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def setup_colab_environment():
    """Set up the Colab environment for DeepSculpt."""
    print("🚀 Setting up DeepSculpt for Google Colab...")
    print("=" * 50)
    
    # Core dependencies that are usually missing in Colab
    # Keep it minimal to avoid version conflicts
    required_packages = [
        "colorama>=0.4.4",
        "rich>=12.0.0",
        "pyyaml>=6.0",
        "h5py>=3.7.0",
        "imageio>=2.22.0",
        "plotly>=5.11.0",
        "tqdm>=4.64.0",
    ]
    
    print("📦 Installing required packages...")
    failed_packages = []
    
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually.")
    else:
        print("\n✅ All packages installed successfully!")
    
    # Check if we're in the right directory
    if os.path.exists("deepsculpt/main.py"):
        print("✅ Found deepsculpt directory")
    else:
        print("❌ deepsculpt directory not found")
        print("Make sure you're in the project root directory")
        return False
    
    # Test import
    print("\n🧪 Testing imports...")
    try:
        sys.path.insert(0, 'deepsculpt')
        from core.models.model_factory import PyTorchModelFactory
        print("✅ DeepSculpt imports working!")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n🎉 Colab setup complete!")
    print("You can now run DeepSculpt commands like:")
    print("  python deepsculpt/main.py --help")
    print("  python deepsculpt/main.py generate-data --num-samples=10")
    
    return True

if __name__ == "__main__":
    setup_colab_environment()