# 🎨 Complete Colab Training Commands for DeepSculpt (FIXED)

## 🚀 Setup First (Run This Once)

```python
# Clone and setup DeepSculpt
!git clone https://github.com/your-org/deepsculpt.git
%cd deepsculpt

# Install dependencies (Colab-optimized)
!pip install colorama rich pyyaml h5py imageio plotly tqdm

# Verify installation
!python deepsculpt/main.py --help
```

## 🎯 GAN Training Commands

### 1. **Quick Test GAN (2-3 minutes)**
```python
# Minimal training for testing (avoid mixed-precision on CPU)
!python deepsculpt/main.py --verbose train-gan \
    --model-type=simple \
    --epochs=2 \
    --batch-size=2 \
    --void-dim=32 \
    --noise-dim=64 \
    --learning-rate=0.0002 \
    --output-dir=./results/quick_test \
    --generate-samples \
    --num-workers=0
```

**Note**: This should now work without visualization errors!

### 2. **Standard GAN Training (15-20 minutes)**
```python
# Balanced training for good results
!python deepsculpt/main.py --verbose train-gan \
    --model-type=skip \
    --epochs=50 \
    --batch-size=8 \
    --void-dim=64 \
    --noise-dim=100 \
    --learning-rate=0.0002 \
    --beta1=0.5 \
    --beta2=0.999 \
    --data-folder=./data \
    --output-dir=./results/gan_standard \
    --snapshot-freq=10 \
    --mixed-precision \
    --gradient-clip=1.0 \
    --scheduler \
    --scheduler-step=20 \
    --scheduler-gamma=0.5 \
    --generate-samples \
    --num-workers=2
```

### 3. **High-Quality GAN Training (30-45 minutes)**
```python
# Full-featured training with all optimizations
!python deepsculpt/main.py --verbose train-gan \
    --model-type=skip \
    --epochs=100 \
    --batch-size=16 \
    --void-dim=64 \
    --noise-dim=100 \
    --learning-rate=0.0002 \
    --beta1=0.5 \
    --beta2=0.999 \
    --data-folder=./data \
    --output-dir=./results/gan_high_quality \
    --snapshot-freq=10 \
    --color \
    --mixed-precision \
    --gradient-clip=1.0 \
    --scheduler \
    --scheduler-step=30 \
    --scheduler-gamma=0.1 \
    --generate-samples \
    --num-workers=2
```

### 4. **Complex GAN Architecture (45-60 minutes)**
```python
# Most advanced GAN training
!python deepsculpt/main.py --verbose train-gan \
    --model-type=complex \
    --epochs=150 \
    --batch-size=12 \
    --void-dim=64 \
    --noise-dim=128 \
    --learning-rate=0.0001 \
    --beta1=0.5 \
    --beta2=0.999 \
    --data-folder=./data \
    --output-dir=./results/gan_complex \
    --snapshot-freq=15 \
    --color \
    --mixed-precision \
    --gradient-clip=0.5 \
    --scheduler \
    --scheduler-step=40 \
    --scheduler-gamma=0.1 \
    --generate-samples \
    --num-workers=2
```

### 5. **Autoencoder GAN (Reconstruction Focus)**
```python
# Autoencoder-based GAN for reconstruction tasks
!python deepsculpt/main.py --verbose train-gan \
    --model-type=autoencoder \
    --epochs=80 \
    --batch-size=8 \
    --void-dim=64 \
    --noise-dim=100 \
    --learning-rate=0.0003 \
    --beta1=0.5 \
    --beta2=0.999 \
    --data-folder=./data \
    --output-dir=./results/gan_autoencoder \
    --snapshot-freq=10 \
    --mixed-precision \
    --gradient-clip=1.0 \
    --scheduler \
    --scheduler-step=25 \
    --scheduler-gamma=0.2 \
    --generate-samples \
    --num-workers=2
```

## 🌊 Diffusion Training Commands

### 1. **Quick Test Diffusion (5-8 minutes)**
```python
# Minimal diffusion training for testing
!python deepsculpt/main.py --verbose train-diffusion \
    --epochs=5 \
    --batch-size=4 \
    --void-dim=32 \
    --timesteps=100 \
    --learning-rate=1e-4 \
    --weight-decay=0.01 \
    --noise-schedule=linear \
    --beta-start=0.0001 \
    --beta-end=0.02 \
    --output-dir=./results/diffusion_quick \
    --mixed-precision \
    --num-workers=2
```

### 2. **Standard Diffusion Training (20-30 minutes)**
```python
# Balanced diffusion training
!python deepsculpt/main.py --verbose train-diffusion \
    --epochs=50 \
    --batch-size=8 \
    --void-dim=64 \
    --timesteps=1000 \
    --learning-rate=1e-4 \
    --weight-decay=0.01 \
    --noise-schedule=cosine \
    --beta-start=0.0001 \
    --beta-end=0.02 \
    --data-folder=./data \
    --output-dir=./results/diffusion_standard \
    --mixed-precision \
    --scheduler \
    --num-workers=2
```

### 3. **High-Quality Diffusion Training (45-60 minutes)**
```python
# Full-featured diffusion training
!python deepsculpt/main.py --verbose train-diffusion \
    --epochs=100 \
    --batch-size=12 \
    --void-dim=64 \
    --timesteps=1000 \
    --learning-rate=5e-5 \
    --weight-decay=0.01 \
    --noise-schedule=cosine \
    --beta-start=0.0001 \
    --beta-end=0.02 \
    --data-folder=./data \
    --output-dir=./results/diffusion_high_quality \
    --mixed-precision \
    --scheduler \
    --num-workers=2
```

### 4. **Advanced Diffusion (Sigmoid Schedule)**
```python
# Advanced diffusion with sigmoid noise schedule
!python deepsculpt/main.py --verbose train-diffusion \
    --epochs=80 \
    --batch-size=10 \
    --void-dim=64 \
    --timesteps=1000 \
    --learning-rate=8e-5 \
    --weight-decay=0.005 \
    --noise-schedule=sigmoid \
    --beta-start=0.00005 \
    --beta-end=0.015 \
    --data-folder=./data \
    --output-dir=./results/diffusion_sigmoid \
    --mixed-precision \
    --scheduler \
    --num-workers=2
```

### 5. **Long Diffusion Training (60-90 minutes)**
```python
# Extended training for best quality
!python deepsculpt/main.py --verbose train-diffusion \
    --epochs=150 \
    --batch-size=8 \
    --void-dim=64 \
    --timesteps=1000 \
    --learning-rate=3e-5 \
    --weight-decay=0.01 \
    --noise-schedule=cosine \
    --beta-start=0.0001 \
    --beta-end=0.02 \
    --data-folder=./data \
    --output-dir=./results/diffusion_extended \
    --mixed-precision \
    --scheduler \
    --num-workers=2
```

## 📊 Data Generation Commands

### Generate Training Data First
```python
# Generate small dataset (quick)
!python deepsculpt/main.py --verbose generate-data \
    --num-samples=100 \
    --void-dim=64 \
    --num-shapes=5 \
    --output-dir=./data/small \
    --sparse-threshold=0.1

# Generate medium dataset (recommended)
!python deepsculpt/main.py --verbose generate-data \
    --num-samples=500 \
    --void-dim=64 \
    --num-shapes=5 \
    --output-dir=./data/medium \
    --sparse-threshold=0.1

# Generate large dataset (for best results)
!python deepsculpt/main.py --verbose generate-data \
    --num-samples=1000 \
    --void-dim=64 \
    --num-shapes=5 \
    --output-dir=./data/large \
    --sparse-threshold=0.1
```

## 🎨 Sample Generation Commands

### After GAN Training
```python
# Generate samples from trained GAN
!python deepsculpt/main.py sample-gan \
    --checkpoint=./results/gan_standard/generator_final.pt \
    --num-samples=10 \
    --output-dir=./samples/gan_output \
    --visualize
```

### After Diffusion Training
```python
# Generate samples from trained diffusion model
!python deepsculpt/main.py sample-diffusion \
    --checkpoint=./results/diffusion_standard/diffusion_final.pt \
    --num-samples=5 \
    --num-steps=50 \
    --output-dir=./samples/diffusion_output \
    --visualize
```

## ⚡ Colab-Optimized Settings

### CPU-Safe Training (No Mixed Precision)
```python
# For CPU or when mixed precision causes issues
!python deepsculpt/main.py --verbose train-gan \
    --model-type=simple \
    --epochs=30 \
    --batch-size=4 \
    --void-dim=32 \
    --noise-dim=64 \
    --num-workers=0
```

### Memory-Efficient Training
```python
# For limited Colab memory (GPU only)
!python deepsculpt/main.py --verbose train-gan \
    --model-type=simple \
    --epochs=30 \
    --batch-size=4 \
    --void-dim=32 \
    --noise-dim=64 \
    --mixed-precision \
    --num-workers=1
```

### GPU-Optimized Training
```python
# When Colab GPU is available
!python deepsculpt/main.py --verbose train-gan \
    --model-type=skip \
    --epochs=100 \
    --batch-size=16 \
    --void-dim=64 \
    --mixed-precision \
    --scheduler \
    --generate-samples
```

## 📈 Monitoring Training

### Check Training Progress
```python
# View training logs
!tail -f ./results/*/training.log

# List all results
!ls -la ./results/

# Check GPU usage (if available)
!nvidia-smi
```

### Visualize Results
```python
# View generated samples
from IPython.display import Image, display
import os

# Display sample images
sample_dir = "./samples/gan_output"
if os.path.exists(sample_dir):
    for img_file in os.listdir(sample_dir):
        if img_file.endswith('.png'):
            display(Image(os.path.join(sample_dir, img_file)))
```

## 🎯 Recommended Training Sequence

```python
# 1. Generate data first
!python deepsculpt/main.py --verbose generate-data --num-samples=500 --void-dim=64

# 2. Train GAN (choose one)
!python deepsculpt/main.py --verbose train-gan --model-type=skip --epochs=50 --batch-size=8 --mixed-precision --generate-samples

# 3. Train Diffusion (choose one)  
!python deepsculpt/main.py --verbose train-diffusion --epochs=50 --batch-size=8 --noise-schedule=cosine --mixed-precision

# 4. Generate samples
!python deepsculpt/main.py sample-gan --checkpoint=./results/gan_*/generator_final.pt --num-samples=10 --visualize
!python deepsculpt/main.py sample-diffusion --checkpoint=./results/diffusion_*/diffusion_final.pt --num-samples=5 --visualize
```

## � Troublesshooting

### Common Issues and Solutions

#### **Channel Mismatch Error**
```
Error: expected input[...] to have 1 channels, but got 64 channels instead
```
**Solution**: Use `simple` model type and avoid mixed precision:
```python
!python deepsculpt/main.py --verbose train-gan \
    --model-type=simple \
    --epochs=10 \
    --batch-size=4 \
    --void-dim=32 \
    --num-workers=0
```

#### **Visualization Error**
```
ValueError: Argument filled must be 3-dimensional
```
**Solution**: This has been fixed! The visualizer now automatically handles tensor dimensions. If you still see this error, update your code.

#### **PyTorch Version Warnings**
```
FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated
```
**Solution**: These are warnings, not errors. Training will continue normally.

#### **CUDA Not Available**
```
Warning: CUDA not available, using CPU
```
**Solution**: Enable GPU in Colab (Runtime → Change runtime type → GPU) or use CPU-optimized settings:
```python
!python deepsculpt/main.py --cpu --verbose train-gan \
    --model-type=simple \
    --epochs=5 \
    --batch-size=2 \
    --void-dim=32 \
    --num-workers=0
```

#### **Out of Memory**
**Solution**: Reduce batch size and void dimension:
```python
!python deepsculpt/main.py --verbose train-gan \
    --model-type=simple \
    --epochs=10 \
    --batch-size=1 \
    --void-dim=16 \
    --num-workers=0
```

## 💡 Pro Tips for Colab

1. **Start with simple model type** to avoid architecture issues
2. **Avoid mixed-precision on CPU** - only use with GPU
3. **Use `--num-workers=0`** to avoid multiprocessing issues
4. **Monitor GPU usage** with `!nvidia-smi`
5. **Save checkpoints frequently** in case of disconnection
6. **Use smaller batch sizes** if you run out of memory
7. **Global arguments** (`--verbose`, `--cpu`) go **before** the command
8. **Command-specific arguments** go **after** the command

## ✅ **Correct Argument Order**

```python
# ✅ CORRECT: Global args before command
!python deepsculpt/main.py --verbose train-gan --epochs=50

# ❌ WRONG: Global args after command  
!python deepsculpt/main.py train-gan --epochs=50 --verbose
```

Choose the commands that fit your time budget and quality requirements! 🚀✨