"""
PyTorch models package for DeepSculpt.

This package contains all PyTorch model implementations including:
- Base model classes and interfaces
- GAN generators and discriminators
- Diffusion models (U-Net, noise schedulers, pipelines)
- Model factory for creating and managing models
"""

from .base_models import (
    BaseGenerator,
    BaseDiscriminator,
    BaseDiffusionModel,
    SparseConv3d,
    SparseConvTranspose3d,
    SparseBatchNorm3d
)

from .model_factory import (
    PyTorchModelFactory,
    create_pytorch_generator,
    create_pytorch_discriminator,
    create_pytorch_diffusion_model,
    create_pytorch_gan_pair
)

# GAN models
from .gan.generator import (
    SimpleGenerator,
    ComplexGenerator,
    SkipGenerator,
    MonochromeGenerator,
    AutoencoderGenerator,
    ProgressiveGenerator,
    ConditionalGenerator
)

from .gan.discriminator import (
    SimpleDiscriminator,
    ComplexDiscriminator,
    ProgressiveDiscriminator,
    ConditionalDiscriminator,
    SpectralNormDiscriminator,
    MultiScaleDiscriminator,
    PatchDiscriminator
)

# Diffusion models
from .diffusion.unet import (
    UNet3D,
    ConditionalUNet3D,
    TimeEmbedding,
    ResBlock3D,
    AttentionBlock3D,
    CrossAttentionBlock3D
)

from .diffusion.noise_scheduler import (
    NoiseScheduler,
    DDIMScheduler,
    DPMSolverScheduler,
    AdaptiveScheduler
)

from .diffusion.pipeline import (
    Diffusion3DPipeline,
    ConditionalDiffusion3DPipeline,
    FastSamplingPipeline,
    ProgressiveDiffusion3DPipeline
)

__all__ = [
    # Base classes
    "BaseGenerator",
    "BaseDiscriminator", 
    "BaseDiffusionModel",
    "SparseConv3d",
    "SparseConvTranspose3d",
    "SparseBatchNorm3d",
    
    # Factory
    "PyTorchModelFactory",
    "create_pytorch_generator",
    "create_pytorch_discriminator",
    "create_pytorch_diffusion_model",
    "create_pytorch_gan_pair",
    
    # GAN Generators
    "SimpleGenerator",
    "ComplexGenerator",
    "SkipGenerator",
    "MonochromeGenerator",
    "AutoencoderGenerator",
    "ProgressiveGenerator",
    "ConditionalGenerator",
    
    # GAN Discriminators
    "SimpleDiscriminator",
    "ComplexDiscriminator",
    "ProgressiveDiscriminator",
    "ConditionalDiscriminator",
    "SpectralNormDiscriminator",
    "MultiScaleDiscriminator",
    "PatchDiscriminator",
    
    # Diffusion Models
    "UNet3D",
    "ConditionalUNet3D",
    "TimeEmbedding",
    "ResBlock3D",
    "AttentionBlock3D",
    "CrossAttentionBlock3D",
    
    # Noise Schedulers
    "NoiseScheduler",
    "DDIMScheduler",
    "DPMSolverScheduler",
    "AdaptiveScheduler",
    
    # Diffusion Pipelines
    "Diffusion3DPipeline",
    "ConditionalDiffusion3DPipeline",
    "FastSamplingPipeline",
    "ProgressiveDiffusion3DPipeline",
]