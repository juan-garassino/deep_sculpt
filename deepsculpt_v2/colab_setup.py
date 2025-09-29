#!/usr/bin/env python3
"""
Google Colab Setup Script for DeepSculpt v2.0

Automated setup and optimization for Google Colab environment with:
- GPU detection and optimization
- Memory management for Colab constraints
- One-command installation and execution
- Colab-specific configurations
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ColabSetup:
    """Google Colab setup and optimization manager."""
    
    def __init__(self):
        self.is_colab = self._detect_colab()
        self.gpu_available = self._check_gpu()
        self.memory_info = self._get_memory_info()
        
    def _detect_colab(self):
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _check_gpu(self):
        """Check GPU availability and type."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✓ GPU detected: {gpu_name}")
                print(f"✓ GPU memory: {gpu_memory:.1f} GB")
                return True
            else:
                print("⚠ No GPU detected - using CPU")
                return False
        except ImportError:
            print("⚠ PyTorch not installed - cannot check GPU")
            return False
    
    def _get_memory_info(self):
        """Get system memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / 1e9,
                'available_gb': memory.available / 1e9,
                'percent_used': memory.percent
            }
        except ImportError:
            return {'total_gb': 12.7, 'available_gb': 10.0, 'percent_used': 20}  # Colab defaults
    
    def install_dependencies(self):
        """Install DeepSculpt v2.0 and dependencies optimized for Colab."""
        print("🚀 Installing DeepSculpt v2.0 for Google Colab...")
        
        # Update pip
        print("📦 Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True, text=True)
        
        # Install core dependencies
        print("📦 Installing core dependencies...")
        core_deps = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.21.0",
            "matplotlib>=3.6.0",
            "plotly>=5.11.0",
            "tqdm>=4.64.0",
            "rich>=12.0.0",
            "pyyaml>=6.0",
            "scikit-learn>=1.1.0"
        ]
        
        for dep in core_deps:
            print(f"  Installing {dep.split('>=')[0]}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    ⚠ Warning: Failed to install {dep}")
        
        # Install Colab-specific dependencies
        print("📦 Installing Colab-specific dependencies...")
        colab_deps = [
            "google-colab",
            "ipywidgets",
            "plotly>=5.0.0",
            "kaleido"  # For plotly static image export
        ]
        
        for dep in colab_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          capture_output=True, text=True)
        
        print("✅ Dependencies installed successfully!")
    
    def setup_colab_environment(self):
        """Setup Colab-specific environment configurations."""
        print("⚙️ Setting up Colab environment...")
        
        # Enable GPU if available
        if self.gpu_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("✓ GPU environment configured")
        
        # Set memory-optimized configurations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Configure matplotlib for Colab
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            print("✓ Matplotlib configured for Colab")
        except ImportError:
            pass
        
        # Setup output directories
        output_dirs = ['./data', './results', './checkpoints', './samples']
        for dir_path in output_dirs:
            Path(dir_path).mkdir(exist_ok=True)
        
        print("✓ Output directories created")
        
        # Create Colab-optimized config
        colab_config = {
            "model": {
                "void_dim": 32 if self.memory_info['total_gb'] < 15 else 64,
                "noise_dim": 100,
                "model_type": "simple"
            },
            "training": {
                "batch_size": 4 if self.memory_info['total_gb'] < 15 else 8,
                "epochs": 10,  # Shorter for Colab
                "learning_rate": 0.0002,
                "mixed_precision": self.gpu_available
            },
            "data": {
                "num_samples": 100,  # Smaller dataset for Colab
                "num_shapes": 3,
                "sparse_threshold": 0.1
            },
            "colab": {
                "gpu_available": self.gpu_available,
                "memory_gb": self.memory_info['total_gb'],
                "optimized_for_colab": True
            }
        }
        
        with open('./colab_config.json', 'w') as f:
            json.dump(colab_config, f, indent=2)
        
        print("✓ Colab-optimized configuration saved")
        print("✅ Colab environment setup complete!")
        
        return colab_config
    
    def optimize_for_colab(self):
        """Apply Colab-specific optimizations."""
        print("🔧 Applying Colab optimizations...")
        
        # Memory optimizations
        if self.gpu_available:
            try:
                import torch
                # Enable memory efficient attention if available
                torch.backends.cuda.enable_flash_sdp(True)
                print("✓ Flash attention enabled")
            except:
                pass
            
            # Set memory fraction to avoid OOM
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)
                print("✓ GPU memory fraction limited to 80%")
            except:
                pass
        
        # CPU optimizations
        try:
            import torch
            torch.set_num_threads(2)  # Limit CPU threads for Colab
            print("✓ CPU threads limited for Colab")
        except:
            pass
        
        print("✅ Colab optimizations applied!")
    
    def create_quick_start_notebook(self):
        """Create a quick start notebook for Colab."""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# DeepSculpt v2.0 - Google Colab Quick Start\n",
                        "\n",
                        "This notebook provides a quick start guide for using DeepSculpt v2.0 in Google Colab.\n",
                        "\n",
                        "## Setup\n",
                        "Run the following cell to install and setup DeepSculpt v2.0:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Install and setup DeepSculpt v2.0\n",
                        "!git clone https://github.com/deepsculpt/deepsculpt.git\n",
                        "%cd deepsculpt/deepSculpt/deepsculpt_v2\n",
                        "\n",
                        "# Run Colab setup\n",
                        "!python colab_setup.py\n",
                        "\n",
                        "# Import required modules\n",
                        "import sys\n",
                        "sys.path.append('.')\n",
                        "\n",
                        "import torch\n",
                        "import json\n",
                        "from pathlib import Path\n",
                        "\n",
                        "print(f\"PyTorch version: {torch.__version__}\")\n",
                        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU: {torch.cuda.get_device_name()}\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Quick Training Example\n",
                        "Train a simple GAN model with Colab-optimized settings:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Quick GAN training\n",
                        "!python main.py train-gan \\\n",
                        "    --model-type=simple \\\n",
                        "    --epochs=5 \\\n",
                        "    --batch-size=4 \\\n",
                        "    --void-dim=32 \\\n",
                        "    --mixed-precision \\\n",
                        "    --generate-samples \\\n",
                        "    --output-dir=./colab_results"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Generate and Visualize Data\n",
                        "Generate synthetic 3D data and create visualizations:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Generate synthetic data\n",
                        "!python main.py generate-data \\\n",
                        "    --num-samples=50 \\\n",
                        "    --void-dim=32 \\\n",
                        "    --sparse \\\n",
                        "    --output-dir=./colab_data\n",
                        "\n",
                        "# Visualize a sample\n",
                        "from core.visualization.pytorch_visualization import PyTorchVisualizer\n",
                        "import torch\n",
                        "\n",
                        "# Load a sample\n",
                        "sample_files = list(Path('./colab_data').glob('*.pt'))\n",
                        "if sample_files:\n",
                        "    sample = torch.load(sample_files[0])\n",
                        "    \n",
                        "    # Create visualizer\n",
                        "    visualizer = PyTorchVisualizer(backend='plotly')\n",
                        "    \n",
                        "    # Plot sculpture\n",
                        "    if isinstance(sample, dict):\n",
                        "        structure = sample.get('structure')\n",
                        "        colors = sample.get('colors')\n",
                        "    else:\n",
                        "        structure = sample\n",
                        "        colors = None\n",
                        "    \n",
                        "    visualizer.plot_sculpture(structure, colors)\n",
                        "    print(\"✓ Visualization created!\")\n",
                        "else:\n",
                        "    print(\"No sample files found\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Diffusion Model Training\n",
                        "Train a diffusion model for 3D generation:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Train diffusion model\n",
                        "!python main.py train-diffusion \\\n",
                        "    --epochs=5 \\\n",
                        "    --batch-size=2 \\\n",
                        "    --void-dim=32 \\\n",
                        "    --timesteps=100 \\\n",
                        "    --mixed-precision \\\n",
                        "    --output-dir=./colab_diffusion"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Performance Benchmarking\n",
                        "Run performance benchmarks to test your Colab setup:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Run benchmarks\n",
                        "!python main.py benchmark \\\n",
                        "    --model-type=simple \\\n",
                        "    --batch-size=4 \\\n",
                        "    --void-dim=32 \\\n",
                        "    --profile-memory \\\n",
                        "    --save-results"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open('./DeepSculpt_v2_Colab_QuickStart.ipynb', 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print("✓ Quick start notebook created: DeepSculpt_v2_Colab_QuickStart.ipynb")
    
    def run_system_check(self):
        """Run comprehensive system check for Colab compatibility."""
        print("🔍 Running system compatibility check...")
        
        checks = []
        
        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(("Python version", "✓", f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python version", "✗", f"{python_version.major}.{python_version.minor} (requires 3.8+)"))
        
        # PyTorch check
        try:
            import torch
            checks.append(("PyTorch", "✓", torch.__version__))
        except ImportError:
            checks.append(("PyTorch", "✗", "Not installed"))
        
        # GPU check
        if self.gpu_available:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                checks.append(("GPU", "✓", gpu_name))
            except:
                checks.append(("GPU", "✗", "Detection failed"))
        else:
            checks.append(("GPU", "⚠", "Not available (CPU only)"))
        
        # Memory check
        total_memory = self.memory_info['total_gb']
        if total_memory >= 10:
            checks.append(("Memory", "✓", f"{total_memory:.1f} GB"))
        else:
            checks.append(("Memory", "⚠", f"{total_memory:.1f} GB (limited)"))
        
        # Disk space check
        try:
            import shutil
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / 1e9
            if free_gb >= 5:
                checks.append(("Disk space", "✓", f"{free_gb:.1f} GB free"))
            else:
                checks.append(("Disk space", "⚠", f"{free_gb:.1f} GB free (limited)"))
        except:
            checks.append(("Disk space", "?", "Cannot determine"))
        
        # Print results
        print("\nSystem Compatibility Report:")
        print("=" * 50)
        for check_name, status, details in checks:
            print(f"{check_name:15} {status} {details}")
        
        # Overall assessment
        passed = sum(1 for _, status, _ in checks if status == "✓")
        total = len(checks)
        
        print("=" * 50)
        if passed >= total - 1:
            print("🎉 System is ready for DeepSculpt v2.0!")
        elif passed >= total - 2:
            print("⚠ System is mostly compatible - some features may be limited")
        else:
            print("❌ System may have compatibility issues")
        
        return checks
    
    def create_colab_optimized_config(self):
        """Create configuration optimized for Colab constraints."""
        print("📝 Creating Colab-optimized configuration...")
        
        # Determine optimal settings based on available resources
        if self.memory_info['total_gb'] >= 15 and self.gpu_available:
            # High-end Colab (Pro/Pro+)
            config = {
                "model": {"void_dim": 64, "batch_size": 8, "model_type": "skip"},
                "training": {"epochs": 20, "mixed_precision": True},
                "data": {"num_samples": 500, "num_shapes": 5}
            }
            tier = "high-end"
        elif self.memory_info['total_gb'] >= 12 and self.gpu_available:
            # Standard Colab with GPU
            config = {
                "model": {"void_dim": 48, "batch_size": 6, "model_type": "simple"},
                "training": {"epochs": 15, "mixed_precision": True},
                "data": {"num_samples": 300, "num_shapes": 4}
            }
            tier = "standard"
        else:
            # Limited Colab (CPU or low memory)
            config = {
                "model": {"void_dim": 32, "batch_size": 4, "model_type": "simple"},
                "training": {"epochs": 10, "mixed_precision": False},
                "data": {"num_samples": 100, "num_shapes": 3}
            }
            tier = "limited"
        
        # Add common settings
        config.update({
            "system": {
                "device": "cuda" if self.gpu_available else "cpu",
                "num_workers": 2,
                "pin_memory": self.gpu_available
            },
            "colab": {
                "tier": tier,
                "gpu_available": self.gpu_available,
                "memory_gb": self.memory_info['total_gb'],
                "optimizations_applied": True
            }
        })
        
        # Save configuration
        with open('./colab_optimized_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration created for {tier} Colab environment")
        return config


def main():
    """Main setup function for Google Colab."""
    print("🎯 DeepSculpt v2.0 - Google Colab Setup")
    print("=" * 50)
    
    # Initialize setup
    setup = ColabSetup()
    
    if not setup.is_colab:
        print("⚠ Warning: Not running in Google Colab")
        print("This script is optimized for Google Colab environment")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run system check
    setup.run_system_check()
    
    # Install dependencies
    setup.install_dependencies()
    
    # Setup environment
    config = setup.setup_colab_environment()
    
    # Apply optimizations
    setup.optimize_for_colab()
    
    # Create optimized configuration
    setup.create_colab_optimized_config()
    
    # Create quick start notebook
    setup.create_quick_start_notebook()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Open 'DeepSculpt_v2_Colab_QuickStart.ipynb' for examples")
    print("2. Use 'colab_optimized_config.json' for optimal settings")
    print("3. Run: python main.py train-gan --config=colab_optimized_config.json")
    print("\nHappy sculpting! 🎨")


if __name__ == "__main__":
    main()