"""
GAN trainer for DeepSculpt PyTorch implementation.

This module provides specialized training infrastructure for GAN models
with advanced training techniques, distributed training support, and
comprehensive monitoring capabilities.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime

from .base_trainer import BaseTrainer, TrainingConfig
from .training_metrics import TrainingMetrics
from .training_loops import GANTrainingLoop


class GANTrainer(BaseTrainer):
    """
    Specialized trainer for GAN models with advanced training techniques.
    
    Features:
    - Adversarial training with generator and discriminator
    - Progressive growing support
    - Curriculum learning
    - Gradient penalty and spectral normalization
    - Advanced loss functions and training strategies
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        gen_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        disc_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        noise_dim: int = 100,
        loss_type: str = "bce",  # "bce", "wgan", "lsgan", "hinge"
        use_gradient_penalty: bool = False,
        gradient_penalty_weight: float = 10.0
    ):
        """
        Initialize GAN trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            gen_optimizer: Generator optimizer
            disc_optimizer: Discriminator optimizer
            config: Training configuration
            gen_scheduler: Generator learning rate scheduler
            disc_scheduler: Discriminator learning rate scheduler
            device: Device for training
            noise_dim: Dimension of noise vector
            loss_type: Type of adversarial loss
            use_gradient_penalty: Whether to use gradient penalty (WGAN-GP)
            gradient_penalty_weight: Weight for gradient penalty
        """
        # Initialize base trainer with generator as main model
        super().__init__(generator, gen_optimizer, config, gen_scheduler, device)
        
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.noise_dim = noise_dim
        self.loss_type = loss_type
        self.use_gradient_penalty = use_gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # Additional scaler for discriminator if using mixed precision
        try:
            # New PyTorch 2.9+ API
            from torch.amp import GradScaler
            self.disc_scaler = GradScaler('cuda') if config.mixed_precision and device != 'cpu' else None
        except ImportError:
            # Fallback for older PyTorch versions
            from torch.cuda.amp import GradScaler
            self.disc_scaler = GradScaler() if config.mixed_precision else None
        
        # GAN-specific metrics
        self.metrics.update({
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'gradient_penalty': [],
            'disc_real_loss': [],
            'disc_fake_loss': []
        })
        
        # Progressive growing parameters
        self.progressive_level = 0
        self.progressive_alpha = 1.0
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_threshold = 0.8
        
        # Training balance parameters
        self.gen_train_freq = 1  # Train generator every N discriminator updates
        self.disc_train_freq = 1  # Train discriminator every N generator updates
        
        # Create fixed noise for consistent evaluation
        self.fixed_noise = torch.randn(16, noise_dim, device=device)
        
        # Setup distributed training for discriminator
        if config.distributed:
            self.discriminator = DDP(self.discriminator, device_ids=[config.rank])
        
        # Initialize training loop
        self.training_loop = GANTrainingLoop(
            generator=self.generator,
            discriminator=self.discriminator,
            loss_type=loss_type,
            use_gradient_penalty=use_gradient_penalty,
            gradient_penalty_weight=gradient_penalty_weight,
            device=device
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = TrainingMetrics()
    
    def adversarial_loss(self, output: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Calculate adversarial loss based on loss type.
        
        Args:
            output: Discriminator output
            target_is_real: Whether target should be real or fake
            
        Returns:
            Computed loss
        """
        if self.loss_type == "bce":
            if target_is_real:
                target = torch.ones_like(output)
            else:
                target = torch.zeros_like(output)
            return F.binary_cross_entropy_with_logits(output, target)
        
        elif self.loss_type == "wgan":
            if target_is_real:
                return -output.mean()
            else:
                return output.mean()
        
        elif self.loss_type == "lsgan":
            if target_is_real:
                target = torch.ones_like(output)
            else:
                target = torch.zeros_like(output)
            return F.mse_loss(output, target)
        
        elif self.loss_type == "hinge":
            if target_is_real:
                return F.relu(1.0 - output).mean()
            else:
                return F.relu(1.0 + output).mean()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated fake samples
            
        Returns:
            Gradient penalty value
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Get discriminator output for interpolates
        disc_interpolates = self.discriminator(interpolates)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            real_data: Batch of real data
            
        Returns:
            Dictionary of step metrics
        """
        batch_size = real_data.size(0)
        
        # Generate noise
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        
        # Use training loop for the actual training step
        step_metrics = self.training_loop.train_step(
            real_data=real_data,
            noise=noise,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            gen_scaler=self.scaler,
            disc_scaler=self.disc_scaler,
            mixed_precision=self.config.mixed_precision,
            gradient_clip=self.config.gradient_clip
        )
        
        # Update metrics tracker
        self.metrics_tracker.update_step_metrics(step_metrics)
        
        return step_metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of epoch metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'gradient_penalty': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                real_data = batch[0].to(self.device)
            elif isinstance(batch, dict):
                # Handle dictionary batch format from StreamingDataset
                structure = batch["structure"].to(self.device)
                
                # Get color mode from discriminator model
                color_mode = getattr(self.discriminator, 'color_mode', 0)
                
                # The actual discriminator expects PyTorch format: [batch, channels, depth, height, width]
                # For monochrome: 1 channel, for color: 6 channels
                if structure.dim() == 4:  # [batch, depth, height, width]
                    if color_mode == 0:  # Monochrome mode expects 1 channel
                        # Add single channel dimension at position 1 (PyTorch format)
                        # [batch, depth, height, width] -> [batch, 1, depth, height, width]
                        real_data = structure.unsqueeze(1)
                    else:  # Color mode expects 6 channels
                        # For color mode, we'd need to create 6 channels from structure and colors
                        colors = batch["colors"].to(self.device)
                        # Create 6 channels - this is a simplified approach
                        real_data = torch.stack([
                            structure, colors, structure, colors, structure, colors
                        ], dim=1)  # Stack along channel dimension
                else:
                    real_data = structure
                
                # Convert to float if needed (models expect float tensors)
                if real_data.dtype != torch.float32:
                    real_data = real_data.float()
            else:
                real_data = batch.to(self.device)
            
            # Progressive growing curriculum
            if self.config.progressive_growing:
                self._update_progressive_growing(batch_idx)
            
            # Curriculum learning
            if self.config.curriculum_learning:
                self._update_curriculum_learning(epoch_metrics)
            
            # Training step
            step_metrics = self.train_step(real_data)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # Log step metrics
            if batch_idx % self.config.log_freq == 0:
                self.log_metrics(step_metrics, self.global_step, "train_step")
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Gen Loss: {step_metrics.get('gen_loss', 0):.4f}, "
                    f"Disc Loss: {step_metrics.get('disc_loss', 0):.4f}"
                )
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items() if values}
        
        # Update learning rate schedulers
        if self.gen_scheduler:
            self.gen_scheduler.step()
        if self.disc_scheduler:
            self.disc_scheduler.step()
        
        # Update metrics tracker
        self.metrics_tracker.update_epoch_metrics(avg_metrics)
        
        return avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    real_data = batch[0].to(self.device)
                else:
                    real_data = batch.to(self.device)
                
                batch_size = real_data.size(0)
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                
                # Generate fake data
                fake_data = self.generator(noise)
                
                # Discriminator outputs
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data)
                
                # Losses
                real_loss = self.adversarial_loss(real_output, True)
                fake_loss = self.adversarial_loss(fake_output, False)
                disc_loss = (real_loss + fake_loss) / 2
                gen_loss = self.adversarial_loss(fake_output, True)
                
                # Accuracies
                if self.loss_type == "bce":
                    real_acc = (torch.sigmoid(real_output) > 0.5).float().mean()
                    fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean()
                else:
                    # For other loss types, use sign-based accuracy
                    real_acc = (real_output > 0).float().mean()
                    fake_acc = (fake_output < 0).float().mean()
                
                val_metrics['gen_loss'].append(gen_loss.item())
                val_metrics['disc_loss'].append(disc_loss.item())
                val_metrics['disc_real_acc'].append(real_acc.item())
                val_metrics['disc_fake_acc'].append(fake_acc.item())
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    def _update_progressive_growing(self, batch_idx: int):
        """Update progressive growing parameters."""
        if hasattr(self.generator, 'grow') and hasattr(self.generator, 'set_alpha'):
            # Simple progressive growing logic
            if batch_idx > 0 and batch_idx % 1000 == 0:  # Grow every 1000 batches
                self.generator.grow()
                if hasattr(self.discriminator, 'grow'):
                    self.discriminator.grow()
                self.progressive_level += 1
                self.logger.info(f"Progressive growing: advanced to level {self.progressive_level}")
    
    def _update_curriculum_learning(self, epoch_metrics: Dict[str, List[float]]):
        """Update curriculum learning parameters."""
        if len(epoch_metrics.get('disc_real_acc', [])) > 10:  # Need some history
            recent_acc = np.mean(epoch_metrics['disc_real_acc'][-10:])
            if recent_acc > self.curriculum_threshold and self.curriculum_stage < 3:
                self.curriculum_stage += 1
                self.logger.info(f"Curriculum learning: advanced to stage {self.curriculum_stage}")
    
    def generate_samples(self, num_samples: int = 16, use_fixed_noise: bool = True) -> torch.Tensor:
        """
        Generate samples for visualization.
        
        Args:
            num_samples: Number of samples to generate
            use_fixed_noise: Whether to use fixed noise for consistency
            
        Returns:
            Generated samples
        """
        self.generator.eval()
        
        with torch.no_grad():
            if use_fixed_noise and num_samples <= len(self.fixed_noise):
                noise = self.fixed_noise[:num_samples]
            else:
                noise = torch.randn(num_samples, self.noise_dim, device=self.device)
            
            samples = self.generator(noise)
        
        return samples
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save GAN training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss,
            'progressive_level': self.progressive_level,
            'curriculum_stage': self.curriculum_stage,
            'loss_type': self.loss_type,
            'use_gradient_penalty': self.use_gradient_penalty,
            'gradient_penalty_weight': self.gradient_penalty_weight
        }
        
        if self.gen_scheduler:
            checkpoint['gen_scheduler_state_dict'] = self.gen_scheduler.state_dict()
        if self.disc_scheduler:
            checkpoint['disc_scheduler_state_dict'] = self.disc_scheduler.state_dict()
        
        if self.scaler:
            checkpoint['gen_scaler_state_dict'] = self.scaler.state_dict()
        if self.disc_scaler:
            checkpoint['disc_scaler_state_dict'] = self.disc_scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"GAN checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load GAN training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        if self.gen_scheduler and 'gen_scheduler_state_dict' in checkpoint:
            self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler_state_dict'])
        if self.disc_scheduler and 'disc_scheduler_state_dict' in checkpoint:
            self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        if self.scaler and 'gen_scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['gen_scaler_state_dict'])
        if self.disc_scaler and 'disc_scaler_state_dict' in checkpoint:
            self.disc_scaler.load_state_dict(checkpoint['disc_scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.progressive_level = checkpoint.get('progressive_level', 0)
        self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
        
        # Update training parameters
        self.loss_type = checkpoint.get('loss_type', self.loss_type)
        self.use_gradient_penalty = checkpoint.get('use_gradient_penalty', self.use_gradient_penalty)
        self.gradient_penalty_weight = checkpoint.get('gradient_penalty_weight', self.gradient_penalty_weight)
        
        self.logger.info(f"GAN checkpoint loaded: {path}")
        return checkpoint
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        info = super().get_training_info()
        info.update({
            "loss_type": self.loss_type,
            "use_gradient_penalty": self.use_gradient_penalty,
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "progressive_level": self.progressive_level,
            "curriculum_stage": self.curriculum_stage,
            "noise_dim": self.noise_dim,
            "generator_params": sum(p.numel() for p in self.generator.parameters()),
            "discriminator_params": sum(p.numel() for p in self.discriminator.parameters()),
        })
        return info