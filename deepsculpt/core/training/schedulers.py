"""
Learning rate schedulers for DeepSculpt PyTorch training.

This module provides advanced learning rate scheduling strategies
optimized for different training scenarios and model types.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List, Union, Callable
import math
import numpy as np
import warnings


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        target_lr: Optional[float] = None,
        last_epoch: int = -1
    ):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            target_lr: Target learning rate (uses base_lr if None)
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            if self.target_lr is not None:
                return [self.target_lr * warmup_factor for _ in self.base_lrs]
            else:
                return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant learning rate after warmup
            if self.target_lr is not None:
                return [self.target_lr for _ in self.base_lrs]
            else:
                return self.base_lrs


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine warmup followed by cosine annealing.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize cosine warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Cosine warmup
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - self.last_epoch / self.warmup_steps)))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class PolynomialScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize polynomial scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            total_steps: Total number of training steps
            power: Polynomial power
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        progress = min(self.last_epoch / self.total_steps, 1.0)
        decay_factor = (1 - progress) ** self.power
        
        return [
            self.min_lr + (base_lr - self.min_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class OneCycleScheduler(_LRScheduler):
    """
    One cycle learning rate policy.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        """
        Initialize one cycle scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing learning rate
            anneal_strategy: Annealing strategy ("cos" or "linear")
            div_factor: Initial learning rate divisor
            final_div_factor: Final learning rate divisor
            last_epoch: Last epoch index
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step_ratio = self.last_epoch / self.total_steps
        
        if step_ratio <= self.pct_start:
            # Increasing phase
            phase_ratio = step_ratio / self.pct_start
            if self.anneal_strategy == "cos":
                lr_factor = (1 - math.cos(math.pi * phase_ratio)) / 2
            else:  # linear
                lr_factor = phase_ratio
            
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * lr_factor
        else:
            # Decreasing phase
            phase_ratio = (step_ratio - self.pct_start) / (1 - self.pct_start)
            if self.anneal_strategy == "cos":
                lr_factor = (1 + math.cos(math.pi * phase_ratio)) / 2
            else:  # linear
                lr_factor = 1 - phase_ratio
            
            lr = self.final_lr + (self.max_lr - self.final_lr) * lr_factor
        
        return [lr for _ in self.base_lrs]


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler (used in Transformer).
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1
    ):
        """
        Initialize Noam scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            model_size: Model dimension (for scaling)
            warmup_steps: Number of warmup steps
            factor: Scaling factor
            last_epoch: Last epoch index
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(self.last_epoch, 1)
        lr = self.factor * (self.model_size ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        return [lr for _ in self.base_lrs]


class CyclicMomentumScheduler:
    """
    Cyclic momentum scheduler to complement cyclic learning rate.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = "triangular"
    ):
        """
        Initialize cyclic momentum scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            base_momentum: Base momentum value
            max_momentum: Maximum momentum value
            step_size_up: Number of steps in increasing half of cycle
            step_size_down: Number of steps in decreasing half of cycle
            mode: Cycling mode
        """
        self.optimizer = optimizer
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.last_epoch = 0
        
        # Check if optimizer supports momentum
        self.has_momentum = hasattr(optimizer.param_groups[0], 'momentum')
        if not self.has_momentum:
            warnings.warn("Optimizer does not support momentum, scheduler will have no effect")
    
    def step(self):
        """Update momentum for current step."""
        if not self.has_momentum:
            return
        
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1 / (2.0 ** (cycle - 1))
        else:
            scale_factor = 1.0
        
        # Momentum cycles inversely to learning rate
        momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * max(0, (1 - x)) * scale_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['momentum'] = momentum
        
        self.last_epoch += 1


class AdaptiveScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler based on training metrics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        last_epoch: int = -1
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            patience: Number of epochs with no improvement after which learning rate will be reduced
            factor: Factor by which the learning rate will be reduced
            threshold: Threshold for measuring the new optimum
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Minimum learning rate
            eps: Minimal decay applied to lr
            last_epoch: Last epoch index
        """
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        
        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.mode = "min"  # Assume we want to minimize the metric
        
        super().__init__(optimizer, last_epoch)
    
    def step(self, metric: float):
        """
        Step the scheduler with a metric value.
        
        Args:
            metric: Current metric value
        """
        current = metric
        
        if self.best_metric is None:
            self.best_metric = current
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self._is_better(current, self.best_metric):
            self.best_metric = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.cooldown_counter == 0 and self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        self.last_epoch += 1
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < best - self.threshold
        else:
            return current > best + self.threshold
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
    
    def get_lr(self):
        """Get current learning rates."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class SchedulerManager:
    """
    Manager for handling multiple schedulers and complex scheduling strategies.
    """
    
    def __init__(self):
        self.schedulers = {}
        self.step_count = 0
    
    def add_scheduler(self, name: str, scheduler: _LRScheduler, step_frequency: str = "epoch"):
        """
        Add a scheduler to the manager.
        
        Args:
            name: Name of the scheduler
            scheduler: Scheduler instance
            step_frequency: When to step ("epoch", "batch", "manual")
        """
        self.schedulers[name] = {
            "scheduler": scheduler,
            "step_frequency": step_frequency,
            "active": True
        }
    
    def step_epoch_schedulers(self, metrics: Optional[Dict[str, float]] = None):
        """Step all epoch-based schedulers."""
        for name, config in self.schedulers.items():
            if config["active"] and config["step_frequency"] == "epoch":
                scheduler = config["scheduler"]
                if isinstance(scheduler, (optim.lr_scheduler.ReduceLROnPlateau, AdaptiveScheduler)):
                    if metrics and "loss" in metrics:
                        scheduler.step(metrics["loss"])
                    else:
                        warnings.warn(f"Scheduler {name} requires metrics but none provided")
                else:
                    scheduler.step()
    
    def step_batch_schedulers(self):
        """Step all batch-based schedulers."""
        for name, config in self.schedulers.items():
            if config["active"] and config["step_frequency"] == "batch":
                config["scheduler"].step()
        self.step_count += 1
    
    def step_scheduler(self, name: str, metrics: Optional[Dict[str, float]] = None):
        """Step a specific scheduler manually."""
        if name in self.schedulers:
            scheduler = self.schedulers[name]["scheduler"]
            if isinstance(scheduler, (optim.lr_scheduler.ReduceLROnPlateau, AdaptiveScheduler)):
                if metrics and "loss" in metrics:
                    scheduler.step(metrics["loss"])
                else:
                    warnings.warn(f"Scheduler {name} requires metrics but none provided")
            else:
                scheduler.step()
    
    def get_current_lrs(self) -> Dict[str, List[float]]:
        """Get current learning rates from all schedulers."""
        lrs = {}
        for name, config in self.schedulers.items():
            if config["active"]:
                scheduler = config["scheduler"]
                if hasattr(scheduler, 'get_last_lr'):
                    lrs[name] = scheduler.get_last_lr()
                else:
                    lrs[name] = [param_group['lr'] for param_group in scheduler.optimizer.param_groups]
        return lrs
    
    def activate_scheduler(self, name: str):
        """Activate a scheduler."""
        if name in self.schedulers:
            self.schedulers[name]["active"] = True
    
    def deactivate_scheduler(self, name: str):
        """Deactivate a scheduler."""
        if name in self.schedulers:
            self.schedulers[name]["active"] = False
    
    def get_scheduler_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all schedulers."""
        info = {}
        for name, config in self.schedulers.items():
            scheduler = config["scheduler"]
            info[name] = {
                "type": scheduler.__class__.__name__,
                "step_frequency": config["step_frequency"],
                "active": config["active"],
                "current_lr": [param_group['lr'] for param_group in scheduler.optimizer.param_groups]
            }
        return info


# Convenience functions for creating common scheduler combinations
def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0
) -> CosineWarmupScheduler:
    """Create a warmup + cosine annealing scheduler."""
    return CosineWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr)


def create_one_cycle_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3
) -> OneCycleScheduler:
    """Create a one cycle scheduler."""
    return OneCycleScheduler(optimizer, max_lr, total_steps, pct_start)


def create_gan_schedulers(
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    scheduler_type: str = "linear_decay",
    **kwargs
) -> tuple[Optional[_LRScheduler], Optional[_LRScheduler]]:
    """
    Create schedulers optimized for GAN training.
    
    Args:
        gen_optimizer: Generator optimizer
        disc_optimizer: Discriminator optimizer
        scheduler_type: Type of scheduler
        **kwargs: Scheduler parameters
        
    Returns:
        Tuple of (generator_scheduler, discriminator_scheduler)
    """
    if scheduler_type == "linear_decay":
        total_steps = kwargs.get("total_steps", 100000)
        gen_scheduler = PolynomialScheduler(gen_optimizer, total_steps, power=1.0)
        disc_scheduler = PolynomialScheduler(disc_optimizer, total_steps, power=1.0)
        return gen_scheduler, disc_scheduler
    
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.5)
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size, gamma)
        disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size, gamma)
        return gen_scheduler, disc_scheduler
    
    else:
        return None, None