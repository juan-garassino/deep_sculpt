# 🚀 Google Colab Setup for DeepSculpt

## Quick Setup (Copy-Paste into Colab)

Run this cell in Google Colab to set up DeepSculpt:

```python
# Install and setup DeepSculpt
!git clone https://github.com/your-org/deepsculpt.git
%cd deepsculpt

# Install only essential dependencies (Colab-optimized)
!pip install -r requirements-colab.txt

# Test the installation
!python deepsculpt/main.py --help
```

## Alternative: Manual dependency installation

```python
# Clone repository
!git clone https://github.com/your-org/deepsculpt.git
%cd deepsculpt

# Install essential packages only
!pip install colorama rich pyyaml h5py imageio plotly tqdm

# Test
!python deepsculpt/main.py --help
```

## Full Installation (if you need all features)

```python
# Clone and install all dependencies
!git clone https://github.com/your-org/deepsculpt.git
%cd deepsculpt
!pip install -r requirements.txt

# Test
!python deepsculpt/main.py --help
```

## Quick Start Examples

Once setup is complete, try these commands:

```python
# Generate sample data
!python deepsculpt/main.py generate-data --num-samples=10 --void-dim=32

# Train a small GAN model
!python deepsculpt/main.py train-gan --model-type=skip --epochs=1 --batch-size=2 --void-dim=32

# Generate samples (after training)
!python deepsculpt/main.py sample-gan --checkpoint=./results/gan_*/generator_final.pt --num-samples=5 --visualize
```

## Troubleshooting

### Error: "No module named 'colorama'"
```python
!pip install colorama rich typer click pyyaml h5py zarr imageio plotly tqdm
```

### Error: "Run from the deepsculpt directory"
Make sure you're in the right directory:
```python
%cd deepsculpt  # If you're in the project root
# OR
%cd /content/deepsculpt  # If you cloned to /content/
```

### Error: "No module named 'core'"
Add the deepsculpt directory to Python path:
```python
import sys
sys.path.insert(0, '/content/deepsculpt')
```

## Memory Optimization for Colab

For Colab's limited memory, use these settings:

```python
# Small model training
!python deepsculpt/main.py train-gan \
    --model-type=simple \
    --epochs=5 \
    --batch-size=2 \
    --void-dim=32 \
    --mixed-precision

# Generate small dataset
!python deepsculpt/main.py generate-data \
    --num-samples=50 \
    --void-dim=32 \
    --output-dir=./small_data
```

## GPU Usage

Colab provides free GPU access. Enable it in Runtime > Change runtime type > GPU.

The system will automatically detect and use GPU when available.