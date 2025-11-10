"""
Training infrastructure package for DeepSculpt PyTorch implementation.

This package contains all training-related components including:
- Base trainer class and configuration
- Specialized trainers for GAN and diffusion models
- Optimizers and learning rate schedulers
- Training loops and metrics tracking
"""

from .base_trainer import BaseTrainer, TrainingConfig
from .gan_trainer import GANTrainer
from .diffusion_trainer import DiffusionTrainer
from .optimizers import (
    OptimizerFactory,
    GradientClipper,
    create_adam_optimizer,
    create_adamw_optimizer
)
from .schedulers import (
    LinearWarmupScheduler,
    CosineWarmupScheduler,
    PolynomialScheduler,
    OneCycleScheduler,
    NoamScheduler,
    CyclicMomentumScheduler,
    AdaptiveScheduler,
    SchedulerManager,
    create_warmup_cosine_scheduler,
    create_one_cycle_scheduler,
    create_gan_schedulers
)
from .training_loops import (
    GANTrainingLoop,
    DiffusionTrainingLoop,
    ProgressiveTrainingLoop,
    CurriculumTrainingLoop,
    create_gan_training_loop,
    create_diffusion_training_loop
)
from .training_metrics import (
    MetricsBuffer,
    TrainingMetrics
)

__all__ = [
    # Base trainer
    "BaseTrainer",
    "TrainingConfig",
    
    # Specialized trainers
    "GANTrainer",
    "DiffusionTrainer",
    
    # Optimizers
    "OptimizerFactory",
    "GradientClipper",
    "create_adam_optimizer",
    "create_adamw_optimizer",
    
    # Schedulers
    "LinearWarmupScheduler",
    "CosineWarmupScheduler",
    "PolynomialScheduler",
    "OneCycleScheduler",
    "NoamScheduler",
    "CyclicMomentumScheduler",
    "AdaptiveScheduler",
    "SchedulerManager",
    "create_warmup_cosine_scheduler",
    "create_one_cycle_scheduler",
    "create_gan_schedulers",
    
    # Training loops
    "GANTrainingLoop",
    "DiffusionTrainingLoop",
    "ProgressiveTrainingLoop",
    "CurriculumTrainingLoop",
    "create_gan_training_loop",
    "create_diffusion_training_loop",
    
    # Metrics
    "MetricsBuffer",
    "TrainingMetrics",
]