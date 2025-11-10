# 🎨 DeepSculpt v2.0 - Complete Pipeline

This directory contains a comprehensive end-to-end pipeline for DeepSculpt v2.0 that handles the complete workflow from data generation to model training and evaluation.

## 🚀 Quick Start

### Option 1: Use the Pipeline Script
```bash
# Quick demo (5-10 minutes)
python pipeline.py --pipeline=demo

# Full GAN pipeline
python pipeline.py --pipeline=complete --model-type=gan --epochs=50

# Diffusion model pipeline
python pipeline.py --pipeline=complete --model-type=diffusion --epochs=20

# Data generation only
python pipeline.py --pipeline=data-only --num-samples=1000
```

### Option 2: Use the Interactive Example
```bash
python examples/complete_pipeline_example.py
```

### Option 3: Use the Main CLI
```bash
# Generate data and train GAN
python main.py generate-data --num-samples=1000 --sparse --output-dir=./data
python main.py train-gan --model-type=skip --epochs=50 --mixed-precision --output-dir=./results

# Generate samples
python main.py sample-gan --checkpoint=./results/generator_final.pt --num-samples=10 --visualize
```

## 📋 Pipeline Stages

The complete pipeline consists of 6 stages:

### 1. 📊 Data Generation
- Creates synthetic 3D sculptures using PyTorchSculptor
- Generates voxel structures and color information
- Supports sparse tensor optimization
- Configurable number of shapes and complexity

### 2. 🔄 Data Preprocessing
- Prepares data for training
- Creates train/validation splits
- Applies encoding transformations
- Optimizes data format for PyTorch

### 3. 🧠 Model Training
- **GAN Training**: Generator + Discriminator with adversarial loss
- **Diffusion Training**: U-Net with noise scheduling
- Mixed precision training support
- Automatic checkpointing and monitoring

### 4. 🎨 Sample Generation
- Generates new sculptures from trained models
- **GAN**: Uses random noise vectors
- **Diffusion**: Uses denoising process
- Saves samples in PyTorch format

### 5. 📈 Model Evaluation
- Calculates performance metrics
- Diversity and quality assessment
- Statistical analysis of generated samples
- Model comparison capabilities

### 6. 🎭 Visualization
- Creates 3D visualizations of sculptures
- Training progress plots
- Interactive and static visualizations
- Multiple backend support (Plotly, Matplotlib)

## ⚙️ Configuration Options

### Model Configuration
```bash
--model-type=gan|diffusion     # Type of model to train
--gan-model-type=skip          # GAN architecture (simple, complex, skip, monochrome)
--void-dim=64                  # 3D voxel space dimension
--noise-dim=100                # Noise vector dimension (GAN only)
--timesteps=1000               # Diffusion timesteps (Diffusion only)
```

### Training Configuration
```bash
--epochs=50                    # Number of training epochs
--batch-size=32                # Training batch size
--learning-rate=0.0002         # Learning rate
--mixed-precision              # Enable mixed precision training
```

### Data Configuration
```bash
--num-samples=1000             # Number of data samples to generate
--num-shapes=5                 # Number of shapes per sculpture
--sparse                       # Enable sparse tensor mode
--sparse-threshold=0.1         # Sparse tensor threshold
```

## 📁 Output Structure

The pipeline creates a structured output directory:

```
pipeline_results/
├── pipeline_YYYYMMDD_HHMMSS/
│   ├── data/                  # Generated training data
│   │   ├── *.pt              # PyTorch tensor files
│   │   ├── generation_metadata.json
│   │   └── data_split.json
│   ├── models/                # Trained models
│   │   ├── checkpoints/       # Training checkpoints
│   │   ├── generator_final.pt # Final GAN generator
│   │   ├── discriminator_final.pt # Final GAN discriminator
│   │   └── diffusion_final.pt # Final diffusion model
│   ├── samples/               # Generated samples
│   │   └── *.pt              # Sample sculptures
│   ├── results/               # Evaluation results
│   │   ├── training_metrics.json
│   │   └── evaluation_metrics.json
│   ├── visualizations/        # 3D visualizations
│   │   ├── *.png             # Sample visualizations
│   │   └── training_metrics.png
│   ├── logs/                  # Training logs
│   ├── pipeline_config.yaml  # Pipeline configuration
│   └── pipeline_summary.json # Execution summary
```

## 🎯 Pipeline Types

### Complete Pipeline
Runs all 6 stages from data generation to visualization:
```bash
python pipeline.py --pipeline=complete --model-type=gan --epochs=50
```

### Demo Pipeline
Quick demonstration with minimal settings:
```bash
python pipeline.py --pipeline=demo
```
- 50 data samples
- 5 training epochs
- 32x32x32 voxel space
- 3 evaluation samples

### Data-Only Pipeline
Generates and preprocesses data without training:
```bash
python pipeline.py --pipeline=data-only --num-samples=1000
```

### Train-Only Pipeline
Assumes data exists and runs training + evaluation:
```bash
python pipeline.py --pipeline=train-only --model-type=diffusion
```

## 🔧 Advanced Features

### Performance Monitoring
- Real-time GPU utilization tracking
- Memory usage optimization
- Training progress monitoring
- Automatic performance regression detection

### Experiment Tracking
- MLflow integration
- Weights & Biases support
- Local experiment logging
- Comprehensive metrics tracking

### Optimization
- Automatic batch size adjustment
- Memory optimization suggestions
- GPU utilization optimization
- Mixed precision training

### Google Colab Support
The pipeline is fully compatible with Google Colab:
```bash
# In Colab
!python colab_setup.py
!python pipeline.py --pipeline=demo
```

## 📊 Example Configurations

### High-Quality GAN Training
```bash
python pipeline.py \
    --pipeline=complete \
    --model-type=gan \
    --gan-model-type=skip \
    --void-dim=64 \
    --epochs=100 \
    --batch-size=32 \
    --num-samples=2000 \
    --mixed-precision \
    --sparse
```

### Fast Diffusion Training
```bash
python pipeline.py \
    --pipeline=complete \
    --model-type=diffusion \
    --void-dim=48 \
    --epochs=30 \
    --batch-size=16 \
    --timesteps=100 \
    --num-samples=1000 \
    --mixed-precision
```

### Memory-Optimized Training
```bash
python pipeline.py \
    --pipeline=complete \
    --model-type=gan \
    --void-dim=32 \
    --batch-size=8 \
    --epochs=50 \
    --sparse \
    --sparse-threshold=0.2
```

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce `--batch-size`
- Reduce `--void-dim`
- Enable `--sparse` mode
- Use `--mixed-precision`

### Slow Training
- Increase `--batch-size` if memory allows
- Enable `--mixed-precision`
- Use GPU if available
- Reduce `--void-dim` for faster iterations

### Poor Results
- Increase `--epochs`
- Increase `--num-samples`
- Try different `--model-type`
- Adjust `--learning-rate`

## 📚 Additional Resources

- **Main CLI**: `python main.py --help`
- **Individual Commands**: See `main.py` for specific operations
- **Colab Setup**: `python colab_setup.py`
- **Performance Analysis**: `python -c "from core.utils.performance_optimizer import analyze_and_optimize; analyze_and_optimize()"`

## 🎉 Getting Started

1. **Quick Test**: `python pipeline.py --pipeline=demo`
2. **Full Training**: `python pipeline.py --pipeline=complete`
3. **Interactive**: `python examples/complete_pipeline_example.py`

The pipeline handles all the complexity of data generation, model training, and evaluation, allowing you to focus on experimenting with different configurations and analyzing results!