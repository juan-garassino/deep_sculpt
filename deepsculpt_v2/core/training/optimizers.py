"""
Optimizers and optimization utilities for DeepSculpt PyTorch training.

This module provides optimizer creation, configuration, and advanced optimization
techniques including learning rate scheduling, gradient clipping, and adaptive methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List, Union, Callable
import math
import warnings


class OptimizerFactory:
    """
    Factory for creating and configuring optimizers for DeepSculpt models.
    
    Provides methods to create various optimizers with appropriate hyperparameters
    for different model types and training scenarios.
    """
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = "adam",
        learning_rate: float = 0.0002,
        weight_decay: float = 0.0,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer for the given model.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer ("adam", "adamw", "sgd", "rmsprop", "adagrad")
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Configured optimizer
        """
        parameters = model.parameters()
        
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                parameters,
                lr=learning_rate,
                betas=kwargs.get("betas", (0.5, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=weight_decay
            )
        
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=weight_decay
            )
        
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=kwargs.get("momentum", 0.9),
                weight_decay=weight_decay,
                nesterov=kwargs.get("nesterov", False)
            )
        
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                parameters,
                lr=learning_rate,
                alpha=kwargs.get("alpha", 0.99),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=weight_decay,
                momentum=kwargs.get("momentum", 0.0)
            )
        
        elif optimizer_type.lower() == "adagrad":
            return optim.Adagrad(
                parameters,
                lr=learning_rate,
                lr_decay=kwargs.get("lr_decay", 0.0),
                weight_decay=weight_decay,
                eps=kwargs.get("eps", 1e-10)
            )
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_gan_optimizers(
        generator: nn.Module,
        discriminator: nn.Module,
        gen_lr: float = 0.0002,
        disc_lr: float = 0.0002,
        optimizer_type: str = "adam",
        **kwargs
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Create optimizers for GAN training.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            gen_lr: Generator learning rate
            disc_lr: Discriminator learning rate
            optimizer_type: Type of optimizer
            **kwargs: Additional optimizer parameters
            
        Returns:
            Tuple of (generator_optimizer, discriminator_optimizer)
        """
        gen_optimizer = OptimizerFactory.create_optimizer(
            generator, optimizer_type, gen_lr, **kwargs
        )
        disc_optimizer = OptimizerFactory.create_optimizer(
            discriminator, optimizer_type, disc_lr, **kwargs
        )
        
        return gen_optimizer, disc_optimizer
    
    @staticmethod
    def create_diffusion_optimizer(
        model: nn.Module,
        learning_rate: float = 1e-4,
        optimizer_type: str = "adamw",
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create optimizer optimized for diffusion model training.
        
        Args:
            model: Diffusion model
            learning_rate: Learning rate
            optimizer_type: Type of optimizer
            **kwargs: Additional optimizer parameters
            
        Returns:
            Configured optimizer
        """
        # Default parameters optimized for diffusion models
        if optimizer_type.lower() == "adamw":
            kwargs.setdefault("betas", (0.9, 0.999))
            kwargs.setdefault("weight_decay", 0.01)
        elif optimizer_type.lower() == "adam":
            kwargs.setdefault("betas", (0.9, 0.999))
            kwargs.setdefault("weight_decay", 0.0)
        
        return OptimizerFactory.create_optimizer(
            model, optimizer_type, learning_rate, **kwargs
        )


class GradientClipper:
    """
    Utility class for gradient clipping with various strategies.
    """
    
    def __init__(self, clip_type: str = "norm", clip_value: float = 1.0):
        """
        Initialize gradient clipper.
        
        Args:
            clip_type: Type of clipping ("norm", "value", "adaptive")
            clip_value: Clipping threshold
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.gradient_history = []
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients of the model.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Gradient norm before clipping
        """
        if self.clip_type == "norm":
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            grad_norm = self._compute_grad_norm(model)
        elif self.clip_type == "adaptive":
            grad_norm = self._adaptive_clip(model)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")
        
        # Track gradient history for adaptive clipping
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        self.gradient_history.append(grad_norm)
        
        # Keep only recent history
        if len(self.gradient_history) > 1000:
            self.gradient_history = self.gradient_history[-1000:]
        
        return grad_norm
    
    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute the gradient norm."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def _adaptive_clip(self, model: nn.Module) -> float:
        """Adaptive gradient clipping based on gradient history."""
        grad_norm = self._compute_grad_norm(model)
        
        if len(self.gradient_history) > 10:
            # Use percentile-based adaptive clipping
            import numpy as np
            percentile_95 = np.percentile(self.gradient_history[-100:], 95)
            adaptive_clip_value = max(self.clip_value, percentile_95 * 1.5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_clip_value)
        else:
            # Fall back to normal clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        
        return grad_norm


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_type: str = "linear",
        last_epoch: int = -1
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            warmup_type: Type of warmup ("linear", "cosine")
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_type = warmup_type
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            if self.warmup_type == "linear":
                warmup_factor = self.last_epoch / self.warmup_steps
            elif self.warmup_type == "cosine":
                warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - self.last_epoch / self.warmup_steps)))
            else:
                warmup_factor = 1.0
            
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class CyclicLRScheduler(_LRScheduler):
    """
    Cyclic learning rate scheduler for improved training dynamics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1
    ):
        """
        Initialize cyclic learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            base_lr: Base learning rate
            max_lr: Maximum learning rate
            step_size_up: Number of steps in increasing half of cycle
            step_size_down: Number of steps in decreasing half of cycle
            mode: Cycling mode ("triangular", "triangular2", "exp_range")
            gamma: Scaling factor for exponential mode
            scale_fn: Custom scaling function
            scale_mode: How to scale ("cycle" or "iterations")
            cycle_momentum: Whether to cycle momentum inversely to learning rate
            base_momentum: Base momentum value
            max_momentum: Maximum momentum value
            last_epoch: Last epoch index
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1 / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale_factor = self.gamma ** self.last_epoch
        else:
            scale_factor = 1.0
        
        if self.scale_fn:
            if self.scale_mode == "cycle":
                scale_factor = self.scale_fn(cycle)
            else:
                scale_factor = self.scale_fn(self.last_epoch)
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        
        return [lr for _ in self.base_lrs]


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers.
    """
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        **kwargs
    ) -> Optional[_LRScheduler]:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            **kwargs: Scheduler-specific parameters
            
        Returns:
            Configured scheduler or None
        """
        if scheduler_type.lower() == "none":
            return None
        
        elif scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1)
            )
        
        elif scheduler_type.lower() == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=kwargs.get("milestones", [30, 60, 90]),
                gamma=kwargs.get("gamma", 0.1)
            )
        
        elif scheduler_type.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.95)
            )
        
        elif scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 100),
                eta_min=kwargs.get("eta_min", 0)
            )
        
        elif scheduler_type.lower() == "cosine_restart":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get("T_0", 10),
                T_mult=kwargs.get("T_mult", 2),
                eta_min=kwargs.get("eta_min", 0)
            )
        
        elif scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.1),
                patience=kwargs.get("patience", 10),
                threshold=kwargs.get("threshold", 1e-4)
            )
        
        elif scheduler_type.lower() == "warmup":
            return WarmupScheduler(
                optimizer,
                warmup_steps=kwargs.get("warmup_steps", 1000),
                total_steps=kwargs.get("total_steps", 10000),
                min_lr=kwargs.get("min_lr", 0.0),
                warmup_type=kwargs.get("warmup_type", "linear")
            )
        
        elif scheduler_type.lower() == "cyclic":
            return CyclicLRScheduler(
                optimizer,
                base_lr=kwargs.get("base_lr", 1e-5),
                max_lr=kwargs.get("max_lr", 1e-3),
                step_size_up=kwargs.get("step_size_up", 2000),
                mode=kwargs.get("mode", "triangular")
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Convenience functions
def create_adam_optimizer(model: nn.Module, lr: float = 0.0002, betas: tuple = (0.5, 0.999)) -> torch.optim.Optimizer:
    """Create Adam optimizer with GAN-friendly defaults."""
    return OptimizerFactory.create_optimizer(model, "adam", lr, betas=betas)


def create_adamw_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Create AdamW optimizer with diffusion-friendly defaults."""
    return OptimizerFactory.create_optimizer(model, "adamw", lr, weight_decay=weight_decay)


def create_cosine_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int) -> _LRScheduler:
    """Create cosine annealing scheduler."""
    return SchedulerFactory.create_scheduler(optimizer, "cosine", T_max=total_epochs)