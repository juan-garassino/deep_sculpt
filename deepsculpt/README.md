# 🎨 DeepSculpt - PyTorch 3D Generative Models

**Generate stunning 3D voxel sculptures using GANs and Diffusion models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# Train a GAN model (1 epoch, small test)
python main.py train-gan --model-type=skip --epochs=1 --batch-size=2 --void-dim=32

# Train a Diffusion model
python main.py train-diffusion --epochs=1 --batch-size=2 --void-dim=32

# Generate samples from trained GAN
python main.py sample-gan --checkpoint=./results/gan_skip_*/generator_final.pt --num-samples=5

# Generate samples from trained Diffusion model
python main.py sample-diffusion --checkpoint=./results/diffusion_*/diffusion_final.pt --num-samples=5
```

## 📁 Output Directory Structure

When you run training, outputs are organized as follows:

```
deepsculpt/
├── results/                           # Training results
│   ├── gan_skip_20231211_120000/      # GAN training run
│   │   ├── checkpoints/               # Model checkpoints during training
│   │   │   └── checkpoint_epoch_*.pth
│   │   ├── snapshots/                 # Generated samples during training
│   │   ├── samples/                   # Final generated samples
│   │   ├── generator_final.pt         # Final generator weights
│   │   ├── discriminator_final.pt     # Final discriminator weights
│   │   └── config.json                # Training configuration
│   │
│   └── diffusion_20231211_130000/     # Diffusion training run
│       ├── checkpoints/
│       │   └── checkpoint_epoch_*.pth
│       └── diffusion_final.pt         # Final diffusion model
│
├── checkpoints/                       # Global checkpoints directory
│   └── checkpoint_epoch_*.pth
│
├── data/                              # Generated training data
│   ├── structures/                    # 3D structure tensors
│   │   └── structure_*.pt
│   ├── colors/                        # Color/material tensors
│   │   └── colors_*.pt
│   └── dataset_metadata.json
│
├── samples/                           # Generated samples output
│   ├── sample_0000.pt                 # PyTorch tensor files
│   ├── sample_0000.png                # Visualization images
│   └── ...
│
└── logs/                              # Training logs
    └── tensorboard/
```

## 🎛️ Hyperparameters Reference

### GAN Training (`train-gan`)

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `--model-type` | `skip` | Generator architecture | `simple`, `complex`, `skip`, `monochrome`, `autoencoder` |
| `--epochs` | `100` | Number of training epochs | Any positive integer |
| `--batch-size` | `32` | Training batch size | 1-128 (depends on GPU memory) |
| `--void-dim` | `64` | 3D voxel space dimension | 16, 32, 64, 128 |
| `--noise-dim` | `100` | Latent noise vector dimension | 50-200 |
| `--learning-rate` | `0.0002` | Adam optimizer learning rate | 0.0001-0.001 |
| `--beta1` | `0.5` | Adam beta1 parameter | 0.0-0.9 |
| `--beta2` | `0.999` | Adam beta2 parameter | 0.9-0.999 |
| `--color` | `False` | Enable color mode (6 channels) | Flag |
| `--sparse` | `False` | Use sparse tensor operations | Flag |
| `--mixed-precision` | `False` | Enable mixed precision training | Flag |
| `--gradient-clip` | `1.0` | Gradient clipping value | 0.0-10.0 |
| `--scheduler` | `False` | Use learning rate scheduler | Flag |
| `--snapshot-freq` | `10` | Save snapshots every N epochs | Any positive integer |
| `--output-dir` | `./results` | Output directory | Any valid path |

### Diffusion Training (`train-diffusion`)

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `--epochs` | `100` | Number of training epochs | Any positive integer |
| `--batch-size` | `16` | Training batch size | 1-64 (diffusion uses more memory) |
| `--void-dim` | `64` | 3D voxel space dimension | 16, 32, 64 |
| `--timesteps` | `1000` | Diffusion timesteps | 100-2000 |
| `--learning-rate` | `0.0001` | AdamW optimizer learning rate | 0.00001-0.001 |
| `--weight-decay` | `0.01` | Weight decay for AdamW | 0.0-0.1 |
| `--noise-schedule` | `linear` | Noise scheduling strategy | `linear`, `cosine`, `sigmoid` |
| `--beta-start` | `0.0001` | Beta schedule start value | 0.00001-0.001 |
| `--beta-end` | `0.02` | Beta schedule end value | 0.01-0.1 |
| `--color` | `False` | Enable color mode (6 channels) | Flag |
| `--sparse` | `False` | Use sparse tensor operations | Flag |
| `--mixed-precision` | `False` | Enable mixed precision training | Flag |

### Data Generation (`generate-data`)

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `--num-samples` | `1000` | Number of samples to generate | Any positive integer |
| `--void-dim` | `64` | 3D voxel space dimension | 16, 32, 64, 128 |
| `--num-shapes` | `5` | Number of shapes per sculpture | 1-20 |
| `--sparse` | `False` | Use sparse tensor format | Flag |
| `--sparse-threshold` | `0.1` | Sparsity threshold | 0.0-1.0 |
| `--output-dir` | `./data` | Output directory | Any valid path |

### Sampling (`sample-gan`, `sample-diffusion`)

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `--checkpoint` | Required | Path to model checkpoint | Valid file path |
| `--num-samples` | `10` | Number of samples to generate | Any positive integer |
| `--output-dir` | `./samples` | Output directory | Any valid path |
| `--visualize` | `False` | Create PNG visualizations | Flag |
| `--num-steps` | `50` | Denoising steps (diffusion only) | 10-1000 |

## 🏗️ Model Architectures

### GAN Generator Types

| Type | Description | Best For | Parameters |
|------|-------------|----------|------------|
| `simple` | Basic 3D transposed convolutions | Quick experiments | ~500K |
| `complex` | Enhanced with batch norm and dropout | Better quality | ~1.2M |
| `skip` | U-Net style skip connections | **Recommended** | ~1.3M |
| `monochrome` | Optimized for single-channel output | Binary structures | ~400K |
| `autoencoder` | Encoder-decoder architecture | Reconstruction | ~2M |

### Discriminator Types

| Type | Description | Use Case |
|------|-------------|----------|
| `simple` | Basic 3D convolutions | Standard training |
| `complex` | Enhanced architecture | Better gradients |
| `progressive` | Progressive growing | High resolution |
| `spectral_norm` | Spectral normalization | Training stability |
| `multi_scale` | Multi-scale discrimination | Feature matching |

### Diffusion Models

| Type | Description | Parameters |
|------|-------------|------------|
| `unet3d` | 3D U-Net with attention | ~50M |
| `conditional_unet3d` | Conditional generation | ~55M |

## 📊 Channel Configuration

The number of channels depends on the color mode:

| Mode | Channels | Description |
|------|----------|-------------|
| Monochrome (`--color` not set) | 1 | Binary structure only |
| Color (`--color` flag) | 6 | Structure + RGB colors |

**Tensor Format (PyTorch):** `[batch, channels, depth, height, width]`

## 💾 Checkpoint Format

### GAN Checkpoints
```python
{
    'epoch': int,
    'generator_state_dict': dict,
    'discriminator_state_dict': dict,
    'gen_optimizer_state_dict': dict,
    'disc_optimizer_state_dict': dict,
    'gen_loss': float,
    'disc_loss': float,
    'config': dict
}
```

### Diffusion Checkpoints
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'noise_scheduler': NoiseScheduler,
    'loss': float,
    'config': {
        'void_dim': int,
        'timesteps': int,
        'noise_schedule': str,
        'in_channels': int,
        'out_channels': int
    }
}
```

## 🖼️ Visualization

### During Training
- Snapshots saved every `--snapshot-freq` epochs
- Loss curves logged to console
- Optional TensorBoard/MLflow integration

### After Training
```bash
# Visualize a saved tensor
python main.py visualize --data-path=./samples/sample_0000.pt --backend=plotly --interactive

# Available backends: matplotlib, plotly, open3d
```

## ⚡ Performance Tips

### GPU Training (Recommended)
```bash
# Enable mixed precision for faster training
python main.py train-gan --model-type=skip --epochs=100 --mixed-precision

# Use sparse tensors for memory efficiency
python main.py train-gan --model-type=skip --epochs=100 --sparse
```

### CPU Training
```bash
# Use smaller dimensions and batch size
python main.py train-gan --model-type=skip --epochs=10 --batch-size=2 --void-dim=32
```

### Memory Optimization
- Reduce `--batch-size` if running out of memory
- Use `--sparse` flag for sparse data
- Reduce `--void-dim` for smaller models
- Enable `--mixed-precision` on compatible GPUs

## 🔧 Configuration Files

### config.yaml
```yaml
model:
  void_dim: 64
  noise_dim: 100

training:
  batch_size: 32
  learning_rate: 0.0002
  epochs: 100

data:
  sparse_threshold: 0.1
  num_workers: 4
```

## 📚 Examples

### Train a high-quality GAN
```bash
python main.py train-gan \
    --model-type=skip \
    --epochs=200 \
    --batch-size=32 \
    --void-dim=64 \
    --learning-rate=0.0002 \
    --mixed-precision \
    --scheduler \
    --snapshot-freq=20
```

### Train a diffusion model
```bash
python main.py train-diffusion \
    --epochs=100 \
    --batch-size=8 \
    --void-dim=32 \
    --timesteps=1000 \
    --noise-schedule=cosine \
    --mixed-precision
```

### Generate training data
```bash
python main.py generate-data \
    --num-samples=5000 \
    --void-dim=64 \
    --num-shapes=5 \
    --output-dir=./data/training
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch-size` or `--void-dim` |
| Training too slow | Enable `--mixed-precision`, use GPU |
| Poor generation quality | Increase `--epochs`, try different `--model-type` |
| NaN losses | Reduce `--learning-rate`, enable `--gradient-clip` |

## � APcI Documentation

### Core Architecture Overview

DeepSculpt follows a modular architecture with clear separation of concerns:

```
deepsculpt/
├── core/
│   ├── data/                    # Data generation and processing
│   │   ├── generation/          # Sculptor & Collector
│   │   ├── transforms/          # Curator & preprocessing
│   │   └── loaders/             # Data loading utilities
│   ├── models/                  # Model architectures
│   │   ├── gan/                 # GAN generators & discriminators
│   │   └── diffusion/           # Diffusion models & schedulers
│   ├── training/                # Training infrastructure
│   │   ├── gan_trainer.py       # GAN training logic
│   │   └── diffusion_trainer.py # Diffusion training logic
│   ├── visualization/           # Plotting and visualization
│   ├── workflow/                # Experiment management
│   └── utils/                   # Utilities and logging
└── main.py                      # CLI entry point
```

### 🎨 Data Generation Pipeline

#### PyTorchSculptor
**Purpose**: Creates individual 3D voxel sculptures with various geometric shapes.

```python
from core.data.generation.pytorch_sculptor import PyTorchSculptor

# Initialize sculptor
sculptor = PyTorchSculptor(
    void_dim=64,           # 3D space dimension
    device="cuda",         # GPU acceleration
    sparse_mode=True       # Memory optimization
)

# Create a sculpture with multiple shapes
sculpture = sculptor.create_sculpture(
    edges=2,               # Number of edge shapes
    planes=1,              # Number of plane shapes  
    pipes=1,               # Number of pipe shapes
    size_range=(0.3, 0.7)  # Shape size variation
)

# Returns: {'structure': tensor, 'colors': tensor}
```

**Key Features**:
- **Shape Types**: Edges, planes, pipes, grids with configurable parameters
- **GPU Acceleration**: Automatic device management and tensor operations
- **Sparse Support**: Memory-efficient sparse tensor mode for large voxel spaces
- **Composition**: Combines multiple shapes into complex sculptures
- **Color/Material**: Optional color and material information generation

#### PyTorchCollector  
**Purpose**: Orchestrates batch generation of sculptures for dataset creation.

```python
from core.data.generation.pytorch_collector import PyTorchCollector

# Configure sculptor parameters
sculptor_config = {
    "void_dim": 64,
    "edges": (2, 0.3, 0.5),    # (count, min_size, max_size)
    "planes": (1, 0.3, 0.5),
    "pipes": (1, 0.3, 0.5)
}

# Initialize collector
collector = PyTorchCollector(
    sculptor_config=sculptor_config,
    output_format="pytorch",        # "pytorch", "hdf5", "zarr"
    sparse_threshold=0.1,          # Sparsity threshold
    device="cuda"
)

# Generate dataset
dataset_paths = collector.create_collection(
    num_samples=1000,              # Number of sculptures
    batch_size=32,                 # Processing batch size
    output_dir="./data"            # Output directory
)
```

**Key Features**:
- **Streaming Generation**: Memory-efficient batch processing
- **Multiple Formats**: PyTorch tensors, HDF5, Zarr support
- **Progress Monitoring**: Real-time generation progress and ETA
- **Memory Management**: Automatic batch size adjustment based on available memory
- **Distributed**: Multi-GPU generation support

#### PyTorchCurator
**Purpose**: Preprocesses and transforms raw sculpture data for training.

```python
from core.data.transforms.pytorch_curator import PyTorchCurator

# Initialize curator
curator = PyTorchCurator(
    encoding_method="one_hot",     # "one_hot", "binary", "rgb", "embedding"
    device="cuda",
    sparse_mode=True
)

# Process dataset
processed_dataset = curator.encode_dataset(
    input_dir="./raw_data",
    output_dir="./processed",
    batch_size=64
)
```

**Encoding Methods**:
- **One-hot**: Multi-class categorical encoding for materials
- **Binary**: Simple binary occupancy encoding
- **RGB**: Color-based encoding for visual materials
- **Embedding**: Learned embeddings for complex material properties

### 🤖 Model Training

#### GAN Training Workflow

```python
from core.training.gan_trainer import GANTrainer
from core.models.model_factory import PyTorchModelFactory

# Create models
factory = PyTorchModelFactory()
generator = factory.create_gan_generator(
    model_type="skip",             # Architecture type
    void_dim=64,                   # 3D space size
    noise_dim=100,                 # Latent dimension
    color_mode=0,                  # 0=mono, 1=color
    sparse=True                    # Sparse tensor support
)

discriminator = factory.create_gan_discriminator(
    model_type="simple",
    void_dim=64,
    color_mode=0,
    sparse=True
)

# Setup trainer
trainer = GANTrainer(
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    config=training_config,
    device="cuda",
    noise_dim=100,
    loss_type="bce"               # "bce", "wgan", "lsgan", "hinge"
)

# Train model
metrics = trainer.train(train_dataloader=data_loader)
```

**Training Features**:
- **Loss Types**: BCE, WGAN, LSGAN, Hinge loss support
- **Gradient Penalty**: WGAN-GP implementation for training stability
- **Progressive Growing**: Gradual resolution increase during training
- **Mixed Precision**: Automatic mixed precision for faster training
- **Distributed**: Multi-GPU training support

#### Diffusion Training Workflow

```python
from core.training.diffusion_trainer import DiffusionTrainer
from core.models.diffusion.unet import UNet3D
from core.models.diffusion.noise_scheduler import NoiseScheduler

# Create diffusion model
model = factory.create_diffusion_model(
    model_type="unet3d",
    void_dim=64,
    in_channels=1,
    out_channels=1,
    timesteps=1000
)

# Create noise scheduler
noise_scheduler = NoiseScheduler(
    schedule_type="cosine",        # "linear", "cosine", "sigmoid"
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)

# Setup trainer
trainer = DiffusionTrainer(
    model=model,
    optimizer=optimizer,
    config=training_config,
    noise_scheduler=noise_scheduler,
    device="cuda"
)

# Train model
metrics = trainer.train(train_dataloader=data_loader)
```

**Diffusion Features**:
- **Noise Schedules**: Linear, cosine, sigmoid scheduling
- **U-Net Architecture**: 3D U-Net with attention mechanisms
- **Conditional Generation**: Support for conditional diffusion
- **DDPM/DDIM**: Multiple sampling strategies

### 📊 Workflow Management

#### PyTorchWorkflowManager
**Purpose**: Orchestrates end-to-end training and evaluation workflows.

```python
from core.workflow.pytorch_workflow import PyTorchWorkflowManager

# Initialize workflow manager
workflow = PyTorchWorkflowManager(
    model_name="deepSculpt",
    data_name="sculptures_v1",
    framework="pytorch"
)

# Execute complete training workflow
results = workflow.run_training_workflow(
    model_type="skip",
    epochs=100,
    batch_size=32,
    data_config=data_config,
    training_config=training_config
)
```

**Workflow Features**:
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Model Versioning**: Automatic model checkpointing and versioning
- **Hyperparameter Tuning**: Grid search and Bayesian optimization
- **Pipeline Orchestration**: Prefect-based workflow management

### 📈 Logging and Monitoring

#### RichLogger
**Purpose**: Provides beautiful console output and structured logging.

```python
from core.utils.logger import RichLogger

# Initialize logger
logger = RichLogger(
    name="DeepSculpt",
    level="INFO",
    console_output=True,
    file_output="./logs/training.log",
    rich_tracebacks=True
)

# Log training progress
logger.info("Starting GAN training")
logger.success("Model checkpoint saved")
logger.warning("GPU memory usage high")
logger.error("Training failed", extra={"epoch": 42})
```

**Logging Features**:
- **Rich Console**: Beautiful colored output with progress bars
- **Structured Logging**: JSON-formatted logs with context
- **Progress Tracking**: Real-time training progress visualization
- **Performance Metrics**: GPU memory, training speed monitoring

### 🎯 Sample Generation

#### GAN Sampling
```python
# Load trained generator
generator.load_state_dict(torch.load("generator_final.pt"))
generator.eval()

# Generate samples
with torch.no_grad():
    noise = torch.randn(10, noise_dim, device="cuda")
    samples = generator(noise)
    
# Save samples
for i, sample in enumerate(samples):
    torch.save(sample.cpu(), f"sample_{i:04d}.pt")
```

#### Diffusion Sampling
```python
from core.models.diffusion.pipeline import Diffusion3DPipeline

# Create pipeline
pipeline = Diffusion3DPipeline(
    model=model,
    noise_scheduler=noise_scheduler,
    device="cuda"
)

# Generate samples with denoising
samples = pipeline.sample(
    shape=(1, 1, 64, 64, 64),
    num_steps=50,                  # Denoising steps
    guidance_scale=7.5             # Classifier-free guidance
)
```

### 🖼️ Visualization and Analysis

#### PyTorchVisualizer
**Purpose**: Creates various visualizations of 3D sculptures.

```python
from core.visualization.pytorch_visualization import PyTorchVisualizer

# Initialize visualizer
visualizer = PyTorchVisualizer(
    backend="plotly",              # "matplotlib", "plotly", "open3d"
    device="cuda"
)

# Create 3D visualization
visualizer.plot_sculpture(
    structure=sample_tensor,
    colors=color_tensor,
    save_path="sculpture.png",
    interactive=True
)

# Create cross-sections
visualizer.plot_sections(
    structure=sample_tensor,
    title="Cross Sections",
    save_path="sections.png"
)

# Create point cloud
points = visualizer.voxel_to_pointcloud(sample_tensor)
visualizer.plot_pointcloud(points, interactive=True)
```

**Visualization Types**:
- **3D Sculptures**: Volumetric rendering with materials
- **Cross-sections**: 2D slices through 3D structures
- **Point Clouds**: Interactive 3D point cloud visualization
- **Animations**: Rotating views and time-lapse generation
- **Comparisons**: Side-by-side model comparison plots

### 🔄 End-to-End Workflow

#### Complete Training Pipeline
```bash
# 1. Generate training data
python main.py generate-data \
    --num-samples=5000 \
    --void-dim=64 \
    --num-shapes=5 \
    --sparse \
    --output-dir=./data

# 2. Train GAN model
python main.py train-gan \
    --model-type=skip \
    --epochs=200 \
    --batch-size=32 \
    --void-dim=64 \
    --mixed-precision \
    --scheduler \
    --mlflow

# 3. Generate samples
python main.py sample-gan \
    --checkpoint=./results/gan_skip_*/generator_final.pt \
    --num-samples=50 \
    --visualize

# 4. Evaluate results
python main.py evaluate \
    --checkpoint=./results/gan_skip_*/generator_final.pt \
    --test-data=./test \
    --metrics=all
```

#### Programmatic API Usage
```python
# Complete workflow in Python
from deepsculpt import DeepSculptV2Main

# Initialize
app = DeepSculptV2Main()

# Generate data
app.generate_data(num_samples=1000, void_dim=64)

# Train model
app.train_gan(
    model_type="skip",
    epochs=100,
    batch_size=32,
    mixed_precision=True
)

# Generate samples
app.sample_gan(
    checkpoint="./results/generator_final.pt",
    num_samples=20,
    visualize=True
)
```

### 🎛️ Configuration Management

#### YAML Configuration
```yaml
# config.yaml
model:
  void_dim: 64
  noise_dim: 100
  model_type: "skip"
  sparse: true

training:
  batch_size: 32
  learning_rate: 0.0002
  epochs: 100
  mixed_precision: true
  gradient_clip: 1.0

data:
  num_samples: 5000
  num_shapes: 5
  sparse_threshold: 0.1
  num_workers: 4

visualization:
  backend: "plotly"
  interactive: true
  save_format: "png"

logging:
  level: "INFO"
  file_output: "./logs/training.log"
  mlflow: true
```

## 📄 License

MIT License - See LICENSE file for details.
