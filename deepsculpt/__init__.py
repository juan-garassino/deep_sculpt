"""
DeepSculpt v2.0 - PyTorch-based 3D Generative Models

This is the modern PyTorch implementation of DeepSculpt with:
- Modular architecture
- Sparse tensor support
- Diffusion models
- Enhanced data streaming
- Comprehensive testing
"""

__version__ = "2.0.0"
__author__ = "DeepSculpt Team"

# Core imports
from .core import models, training, data, visualization, workflow, utils

__all__ = [
    "models",
    "training", 
    "data",
    "visualization",
    "workflow",
    "utils"
]