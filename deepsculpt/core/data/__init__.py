"""
Data pipeline package for DeepSculpt PyTorch implementation.

This package contains all data-related components including:
- Data generation and streaming
- Preprocessing and encoding
- Data loaders and augmentations
"""

# Generation components
from .generation.data_generator import (
    DataGenerator,
    ParametricDataGenerator,
    ConditionalDataGenerator,
    create_simple_generator,
    create_parametric_generator
)

from .generation.dataset_streamer import (
    StreamingDataset,
    FileBasedDataset,
    DistributedDataStreamer,
    DatasetSplitter,
    create_streaming_dataloader,
    create_file_dataloader
)

# Transform components
from .transforms.preprocessing import (
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

from .transforms.encoding import (
    BaseEncoder,
    OneHotEncoder,
    BinaryEncoder,
    RGBEncoder,
    LearnedEmbeddingEncoder,
    CompositeEncoder,
    create_standard_encoder,
    create_sculpture_encoder
)

from .transforms.augmentations import (
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

# Loader components
from .loaders.data_loaders import (
    AdaptiveBatchSampler,
    BalancedSampler,
    MultiFormatDataLoader,
    CombinedDataset,
    StreamingDataLoader,
    create_training_dataloader,
    create_validation_dataloader,
    create_streaming_dataloader,
    create_balanced_dataloader
)

__all__ = [
    # Generation
    "DataGenerator",
    "ParametricDataGenerator", 
    "ConditionalDataGenerator",
    "create_simple_generator",
    "create_parametric_generator",
    "StreamingDataset",
    "FileBasedDataset",
    "DistributedDataStreamer",
    "DatasetSplitter",
    
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
    
    # Loaders
    "AdaptiveBatchSampler",
    "BalancedSampler",
    "MultiFormatDataLoader",
    "CombinedDataset",
    "StreamingDataLoader",
    "create_training_dataloader",
    "create_validation_dataloader",
    "create_streaming_dataloader",
    "create_balanced_dataloader",
]