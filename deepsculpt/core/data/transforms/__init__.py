"""
Data transforms package for DeepSculpt PyTorch implementation.

This package contains components for data preprocessing, encoding,
and augmentation of 3D sculpture data.
"""

from .preprocessing import (
    BasePreprocessor,
    NormalizationPreprocessor,
    AugmentationPreprocessor,
    FilteringPreprocessor,
    CompositePreprocessor,
    ConditionalPreprocessor,
    create_standard_preprocessor,
    create_training_preprocessor,
    create_validation_preprocessor
)

from .encoding import (
    BaseEncoder,
    OneHotEncoder,
    BinaryEncoder,
    RGBEncoder,
    LearnedEmbeddingEncoder,
    CompositeEncoder,
    create_standard_encoder,
    create_sculpture_encoder
)

from .augmentations import (
    BaseAugmentation,
    RotationAugmentation,
    ScalingAugmentation,
    NoiseAugmentation,
    FlipAugmentation,
    ElasticDeformationAugmentation,
    CompositeAugmentation,
    create_standard_augmentation_pipeline,
    create_training_augmentation_pipeline,
    create_geometric_augmentation_pipeline
)

__all__ = [
    # Preprocessing
    "BasePreprocessor",
    "NormalizationPreprocessor",
    "AugmentationPreprocessor",
    "FilteringPreprocessor",
    "CompositePreprocessor",
    "ConditionalPreprocessor",
    "create_standard_preprocessor",
    "create_training_preprocessor",
    "create_validation_preprocessor",
    
    # Encoding
    "BaseEncoder",
    "OneHotEncoder",
    "BinaryEncoder",
    "RGBEncoder",
    "LearnedEmbeddingEncoder",
    "CompositeEncoder",
    "create_standard_encoder",
    "create_sculpture_encoder",
    
    # Augmentations
    "BaseAugmentation",
    "RotationAugmentation",
    "ScalingAugmentation",
    "NoiseAugmentation",
    "FlipAugmentation",
    "ElasticDeformationAugmentation",
    "CompositeAugmentation",
    "create_standard_augmentation_pipeline",
    "create_training_augmentation_pipeline",
    "create_geometric_augmentation_pipeline",
]