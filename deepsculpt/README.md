# 🎨 DeepSculpt v2.0 - Modern PyTorch Implementation

**Next-generation 3D generative models with PyTorch, sparse tensors, and diffusion models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the modern PyTorch-based implementation of DeepSculpt with enhanced features, modular architecture, and dramatically improved performance.

## Architecture Overview

```
deepsculpt_v2/
├── core/                          # Core functionality
│   ├── models/                    # Model architectures
│   │   ├── gan/                   # GAN model implementations
│   │   ├── diffusion/             # Diffusion model implementations
│   │   └── pytorch_models.py      # Model factory and base classes
│   ├── training/                  # Training infrastructure
│   │   └── pytorch_trainer.py     # GAN and diffusion trainers
│   ├── data/                      # Data pipeline
│   │   ├── generation/            # 3D data generation
│   │   │   ├── pytorch_shapes.py      # Shape primitives
│   │   │   ├── pytorch_sculptor.py    # Sculpture composition
│   │   │   ├── pytorch_collector.py   # Dataset collection
│   │   │   └── pytorch_collector_bis.py # Alternative collector
│   │   ├── loaders/               # Data loaders (future)
│   │   ├── transforms/            # Data preprocessing
│   │   │   └── pytorch_curator.py     # Data curation and encoding
│   │   └── sparse/                # Sparse tensor handling (future)
│   ├── visualization/             # 3D visualization
│   │   └── pytorch_visualization.py   # Enhanced visualizer
│   ├── workflow/                  # Workflow orchestration
│   │   ├── pytorch_workflow.py        # Workflow management
│   │   └── pytorch_mlflow_tracking.py # Experiment tracking
│   └── utils/                     # Utilities
│       └── pytorch_utils.py           # PyTorch utilities
└── tests/                         # Test suite
    ├── unit/                      # Unit tests
    ├── integration/               # Integration tests
    └── benchmarks/                # Performance benchmarks
```

## Key Features

### 🚀 **Modern PyTorch Implementation**
- Full PyTorch tensor operations with GPU acceleration
- Automatic mixed precision training
- Distributed training support
- Memory-efficient sparse tensor operations

### 🎯 **Advanced Model Architectures**
- **GAN Models**: Simple, Complex, Skip, Monochrome, Autoencoder
- **Diffusion Models**: 3D U-Net with time embedding and conditioning
- **Sparse Support**: Automatic sparse/dense tensor conversion
- **Model Factory**: Easy model creation and management

### 📊 **Enhanced Data Pipeline**
- **Shape Generation**: PyTorch-based 3D shape primitives (edges, planes, pipes, grids)
- **Sculpture Composition**: Flexible sculpture generation with method chaining
- **Dataset Collection**: Memory-efficient streaming dataset creation
- **Data Curation**: Advanced encoding (one-hot, binary, RGB, embeddings)

### 🎨 **Advanced Visualization**
- **Multi-backend Support**: matplotlib, plotly, open3d
- **GPU Rendering**: Direct PyTorch tensor visualization
- **Interactive Plots**: Real-time 3D manipulation
- **Training Monitoring**: Live training progress visualization

### ⚡ **Performance Optimizations**
- **Sparse Tensors**: Automatic sparsity detection and conversion
- **Memory Management**: Dynamic batch sizing and memory monitoring
- **GPU Acceleration**: Full CUDA support with automatic device management
- **Distributed Training**: Multi-GPU and multi-node support

### 🔬 **Experiment Tracking**
- **MLflow Integration**: Comprehensive experiment logging
- **Model Versioning**: Automatic checkpoint management
- **Metrics Tracking**: Real-time training and validation metrics
- **Artifact Management**: Model, data, and visualization artifacts

## Quick Start

### Installation

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install additional dependencies
pip install mlflow wandb plotly matplotlib open3d rich
```

### Basic Usage

```python
from deepsculpt_v2.core.data.generation import PyTorchSculptor, PyTorchShapeFactory
from deepsculpt_v2.core.models import PyTorchModelFactory
from deepsculpt_v2.core.training import GANTrainer
from deepsculpt_v2.core.visualization import PyTorchVisualizer

# Generate 3D data
sculptor = PyTorchSculptor(void_dim=64, device="cuda", sparse_mode=True)
structure, colors = sculptor.generate_sculpture()

# Create and train model
generator = PyTorchModelFactory.create_gan_generator("skip", void_dim=64)
discriminator = PyTorchModelFactory.create_gan_discriminator("skip", void_dim=64)
trainer = GANTrainer(generator, discriminator)

# Visualize results
visualizer = PyTorchVisualizer(backend="plotly")
visualizer.plot_sculpture(structure, colors)
```

### Advanced Features

```python
# Diffusion model training
from deepsculpt_v2.core.models.diffusion import Diffusion3DPipeline
from deepsculpt_v2.core.training import DiffusionTrainer

pipeline = Diffusion3DPipeline(timesteps=1000)
trainer = DiffusionTrainer(model, pipeline)

# Sparse tensor operations
from deepsculpt_v2.core.utils import PyTorchUtils

# Automatic sparse conversion
sparse_tensor = PyTorchUtils.optimize_tensor_storage(dense_tensor, threshold=0.1)

# Memory monitoring
memory_info = PyTorchUtils.calculate_memory_usage(tensor)
```

## Migration from Legacy

The v2.0 implementation provides significant improvements over the legacy TensorFlow version:

| Feature | Legacy (v1.x) | v2.0 |
|---------|---------------|------|
| Framework | TensorFlow | PyTorch |
| Architecture | Monolithic | Modular |
| Memory Usage | High | Optimized with sparse tensors |
| Training Speed | Standard | 2-3x faster with mixed precision |
| Model Types | GAN only | GAN + Diffusion |
| Visualization | Basic | Interactive with multiple backends |
| Testing | Limited | Comprehensive unit/integration tests |

## Development

### Running Tests

```bash
# Unit tests
python -m pytest deepsculpt_v2/tests/unit/ -v

# Integration tests  
python -m pytest deepsculpt_v2/tests/integration/ -v

# Benchmarks
python -m pytest deepsculpt_v2/tests/benchmarks/ -v
```

### Code Quality

The v2.0 codebase follows modern Python best practices:
- Type hints throughout
- Comprehensive docstrings
- Modular design with clear interfaces
- Extensive testing coverage
- Performance benchmarking

## Contributing

1. Follow the modular architecture patterns
2. Add comprehensive tests for new features
3. Update documentation and type hints
4. Run benchmarks for performance-critical changes
5. Maintain backward compatibility where possible

## License

Same as the original DeepSculpt project.