"""
PyTorch training infrastructure for DeepSculpt models.
This module provides comprehensive training classes for GAN and diffusion models
with distributed training, mixed precision, and advanced training techniques.
"""

import os
import time
import math
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 0.0002
    epochs: int = 100
    
    # Optimizer parameters
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Training techniques
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    progressive_growing: bool = False
    curriculum_learning: bool = False
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Checkpointing and logging
    checkpoint_freq: int = 5
    log_freq: int = 10
    snapshot_freq: int = 1
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    snapshot_dir: str = "./snapshots"
    
    # Experiment tracking
    use_wandb: bool = False
    use_mlflow: bool = False
    use_tensorboard: bool = True
    experiment_name: str = "deepsculpt_experiment"


class BaseTrainer:
    """Base class for all trainers with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Current training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup logging and experiment tracking
        self._setup_logging()
        self._setup_experiment_tracking()
        
        # Setup distributed training if enabled
        if config.distributed:
            self._setup_distributed()
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup TensorBoard
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.config.log_dir)
        else:
            self.writer = None
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking (Wandb, MLflow)."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="deepsculpt",
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            mlflow.log_params(self.config.__dict__)
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.model = DDP(self.model, device_ids=[self.config.rank])
        self.logger.info(f"Distributed training setup complete. Rank: {self.config.rank}")
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all configured tracking systems."""
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb_metrics = {f"{prefix}/{key}" if prefix else key: value for key, value in metrics.items()}
            wandb.log(wandb_metrics, step=step)
        
        # MLflow
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(f"{prefix}_{key}" if prefix else key, value, step=step)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch. To be implemented by subclasses."""
        raise NotImplementedError
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model. To be implemented by subclasses."""
        raise NotImplementedError


class GANTrainer(BaseTrainer):
    """Specialized trainer for GAN models with advanced training techniques."""
    
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
        noise_dim: int = 100
    ):
        # Initialize base trainer with generator as main model
        super().__init__(generator, gen_optimizer, config, gen_scheduler, device)
        
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.noise_dim = noise_dim
        
        # Additional scaler for discriminator if using mixed precision
        self.disc_scaler = GradScaler() if config.mixed_precision else None
        
        # GAN-specific metrics
        self.metrics.update({
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'gradient_penalty': []
        })
        
        # Progressive growing parameters
        self.progressive_level = 0
        self.progressive_alpha = 1.0
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_threshold = 0.8
        
        # Create fixed noise for consistent evaluation
        self.fixed_noise = torch.randn(16, noise_dim, device=device)
        
        # Setup distributed training for discriminator
        if config.distributed:
            self.discriminator = DDP(self.discriminator, device_ids=[config.rank])
    
    def adversarial_loss(self, output: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate adversarial loss (binary cross entropy)."""
        if target_is_real:
            target = torch.ones_like(output)
        else:
            target = torch.zeros_like(output)
        
        return F.binary_cross_entropy_with_logits(output, target)
    
    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP."""
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
        """Execute a single training step."""
        batch_size = real_data.size(0)
        
        # Generate noise
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        if self.config.mixed_precision:
            with autocast():
                # Real data
                real_output = self.discriminator(real_data)
                real_loss = self.adversarial_loss(real_output, True)
                
                # Fake data
                with torch.no_grad():
                    fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data.detach())
                fake_loss = self.adversarial_loss(fake_output, False)
                
                # Total discriminator loss
                disc_loss = (real_loss + fake_loss) / 2
                
                # Add gradient penalty if using WGAN-GP
                if hasattr(self.config, 'use_gradient_penalty') and self.config.use_gradient_penalty:
                    gp = self.gradient_penalty(real_data, fake_data)
                    disc_loss += 10.0 * gp
            
            self.disc_scaler.scale(disc_loss).backward()
            
            if self.config.gradient_clip > 0:
                self.disc_scaler.unscale_(self.disc_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip)
            
            self.disc_scaler.step(self.disc_optimizer)
            self.disc_scaler.update()
        else:
            # Real data
            real_output = self.discriminator(real_data)
            real_loss = self.adversarial_loss(real_output, True)
            
            # Fake data
            with torch.no_grad():
                fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data.detach())
            fake_loss = self.adversarial_loss(fake_output, False)
            
            # Total discriminator loss
            disc_loss = (real_loss + fake_loss) / 2
            
            disc_loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip)
            
            self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        if self.config.mixed_precision:
            with autocast():
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data)
                gen_loss = self.adversarial_loss(fake_output, True)
            
            self.scaler.scale(gen_loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.gen_optimizer)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.gradient_clip)
            
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
        else:
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data)
            gen_loss = self.adversarial_loss(fake_output, True)
            
            gen_loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.gradient_clip)
            
            self.gen_optimizer.step()
        
        # Calculate accuracies
        with torch.no_grad():
            real_acc = (torch.sigmoid(real_output) > 0.5).float().mean()
            fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean()
        
        return {
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item(),
            'disc_real_acc': real_acc.item(),
            'disc_fake_acc': fake_acc.item()
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                real_data = batch[0].to(self.device)
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
                epoch_metrics[key].append(value)
            
            # Log step metrics
            if batch_idx % self.config.log_freq == 0:
                self.log_metrics(step_metrics, self.global_step, "train_step")
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Gen Loss: {step_metrics['gen_loss']:.4f}, "
                    f"Disc Loss: {step_metrics['disc_loss']:.4f}"
                )
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Update learning rate schedulers
        if self.gen_scheduler:
            self.gen_scheduler.step()
        if self.disc_scheduler:
            self.disc_scheduler.step()
        
        return avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
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
                real_acc = (torch.sigmoid(real_output) > 0.5).float().mean()
                fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean()
                
                val_metrics['gen_loss'].append(gen_loss.item())
                val_metrics['disc_loss'].append(disc_loss.item())
                val_metrics['disc_real_acc'].append(real_acc.item())
                val_metrics['disc_fake_acc'].append(fake_acc.item())
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    def _update_progressive_growing(self, batch_idx: int):
        """Update progressive growing parameters."""
        if hasattr(self.generator, 'grow') and hasattr(self.generator, 'set_alpha'):
            # Simple progressive growing logic - can be made more sophisticated
            total_batches = len(self.metrics.get('gen_loss', [0])) * 1000  # Rough estimate
            if batch_idx > 0 and batch_idx % 1000 == 0:  # Grow every 1000 batches
                self.generator.grow()
                self.discriminator.grow() if hasattr(self.discriminator, 'grow') else None
                self.progressive_level += 1
                self.logger.info(f"Progressive growing: advanced to level {self.progressive_level}")
    
    def _update_curriculum_learning(self, epoch_metrics: Dict[str, List[float]]):
        """Update curriculum learning parameters."""
        if len(epoch_metrics['disc_real_acc']) > 10:  # Need some history
            recent_acc = np.mean(epoch_metrics['disc_real_acc'][-10:])
            if recent_acc > self.curriculum_threshold and self.curriculum_stage < 3:
                self.curriculum_stage += 1
                self.logger.info(f"Curriculum learning: advanced to stage {self.curriculum_stage}")
    
    def generate_samples(self, num_samples: int = 16, use_fixed_noise: bool = True) -> torch.Tensor:
        """Generate samples for visualization."""
        self.generator.eval()
        
        with torch.no_grad():
            if use_fixed_noise and num_samples <= len(self.fixed_noise):
                noise = self.fixed_noise[:num_samples]
            else:
                noise = torch.randn(num_samples, self.noise_dim, device=self.device)
            
            samples = self.generator(noise)
        
        return samples
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save GAN training checkpoint."""
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
            'curriculum_stage': self.curriculum_stage
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
        """Load GAN training checkpoint."""
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
        
        self.logger.info(f"GAN checkpoint loaded: {path}")
        return checkpoint


class NoiseScheduler:
    """Handles noise scheduling for diffusion process."""
    
    def __init__(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.schedule_type = schedule_type
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Create noise schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(timesteps, device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, device: str) -> torch.Tensor:
        """Create cosine beta schedule."""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples according to the noise schedule."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get velocity for v-parameterization."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class DiffusionTrainer(BaseTrainer):
    """Specialized trainer for diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        noise_scheduler: NoiseScheduler,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        prediction_type: str = "epsilon",  # "epsilon", "sample", "v_prediction"
        conditioning_key: Optional[str] = None
    ):
        super().__init__(model, optimizer, config, scheduler, device)
        
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.conditioning_key = conditioning_key
        
        # Diffusion-specific metrics
        self.metrics.update({
            'diffusion_loss': [],
            'mse_loss': [],
            'l1_loss': [],
            'perceptual_loss': []
        })
        
        # EMA model for better sample quality
        self.ema_model = self._create_ema_model() if hasattr(config, 'use_ema') and config.use_ema else None
        self.ema_decay = getattr(config, 'ema_decay', 0.9999)
        
        self.logger.info(f"Diffusion trainer initialized with {prediction_type} prediction")
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA version of the model."""
        ema_model = type(self.model)(
            **{k: v for k, v in self.model.__dict__.items() 
               if not k.startswith('_') and k != 'training'}
        )
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute diffusion loss based on prediction type."""
        losses = {}
        
        if self.prediction_type == "epsilon":
            # Predict noise
            losses['mse_loss'] = F.mse_loss(model_output, target)
            losses['l1_loss'] = F.l1_loss(model_output, target)
        elif self.prediction_type == "sample":
            # Predict original sample
            losses['mse_loss'] = F.mse_loss(model_output, target)
            losses['l1_loss'] = F.l1_loss(model_output, target)
        elif self.prediction_type == "v_prediction":
            # Predict velocity
            velocity_target = self.noise_scheduler.get_velocity(sample, target, timesteps)
            losses['mse_loss'] = F.mse_loss(model_output, velocity_target)
            losses['l1_loss'] = F.l1_loss(model_output, velocity_target)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Combined loss
        losses['diffusion_loss'] = losses['mse_loss'] + 0.1 * losses['l1_loss']
        
        return losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        # Get clean samples
        if isinstance(batch, dict):
            clean_samples = batch['samples'] if 'samples' in batch else batch['images']
            conditioning = batch.get(self.conditioning_key) if self.conditioning_key else None
        else:
            clean_samples = batch
            conditioning = None
        
        batch_size = clean_samples.size(0)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.timesteps, (batch_size,), device=self.device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(clean_samples)
        
        # Add noise to clean samples
        noisy_samples = self.noise_scheduler.add_noise(clean_samples, noise, timesteps)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if self.config.mixed_precision:
            with autocast():
                if conditioning is not None:
                    model_output = self.model(noisy_samples, timesteps, conditioning)
                else:
                    model_output = self.model(noisy_samples, timesteps)
                
                # Compute losses
                if self.prediction_type == "epsilon":
                    target = noise
                elif self.prediction_type == "sample":
                    target = clean_samples
                else:  # v_prediction
                    target = noise  # Will be converted to velocity in compute_loss
                
                losses = self.compute_loss(model_output, target, timesteps, clean_samples)
                loss = losses['diffusion_loss']
            
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if conditioning is not None:
                model_output = self.model(noisy_samples, timesteps, conditioning)
            else:
                model_output = self.model(noisy_samples, timesteps)
            
            # Compute losses
            if self.prediction_type == "epsilon":
                target = noise
            elif self.prediction_type == "sample":
                target = clean_samples
            else:  # v_prediction
                target = noise  # Will be converted to velocity in compute_loss
            
            losses = self.compute_loss(model_output, target, timesteps, clean_samples)
            loss = losses['diffusion_loss']
            
            loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
        
        # Update EMA model
        self._update_ema_model()
        
        return {key: value.item() for key, value in losses.items()}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'diffusion_loss': [],
            'mse_loss': [],
            'l1_loss': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # Log step metrics
            if batch_idx % self.config.log_freq == 0:
                self.log_metrics(step_metrics, self.global_step, "train_step")
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Diffusion Loss: {step_metrics['diffusion_loss']:.4f}, "
                    f"MSE Loss: {step_metrics['mse_loss']:.4f}"
                )
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Update learning rate scheduler
        if self.scheduler:
            self.scheduler.step()
        
        return avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        model_to_eval = self.ema_model if self.ema_model else self.model
        model_to_eval.eval()
        
        val_metrics = {
            'diffusion_loss': [],
            'mse_loss': [],
            'l1_loss': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    clean_samples = batch['samples'] if 'samples' in batch else batch['images']
                    conditioning = batch.get(self.conditioning_key) if self.conditioning_key else None
                else:
                    batch = batch.to(self.device)
                    clean_samples = batch
                    conditioning = None
                
                batch_size = clean_samples.size(0)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.timesteps, (batch_size,), device=self.device
                ).long()
                
                # Sample noise
                noise = torch.randn_like(clean_samples)
                
                # Add noise to clean samples
                noisy_samples = self.noise_scheduler.add_noise(clean_samples, noise, timesteps)
                
                # Forward pass
                if conditioning is not None:
                    model_output = model_to_eval(noisy_samples, timesteps, conditioning)
                else:
                    model_output = model_to_eval(noisy_samples, timesteps)
                
                # Compute losses
                if self.prediction_type == "epsilon":
                    target = noise
                elif self.prediction_type == "sample":
                    target = clean_samples
                else:  # v_prediction
                    target = noise
                
                losses = self.compute_loss(model_output, target, timesteps, clean_samples)
                
                for key, value in losses.items():
                    if key in val_metrics:
                        val_metrics[key].append(value.item())
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        conditioning: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generate samples using DDPM sampling."""
        model_to_sample = self.ema_model if self.ema_model else self.model
        model_to_sample.eval()
        
        # Create random noise
        sample = torch.randn(shape, device=self.device, generator=generator)
        
        # Create timesteps for inference
        timesteps = torch.linspace(
            self.noise_scheduler.timesteps - 1, 0, num_inference_steps, device=self.device
        ).long()
        
        for i, t in enumerate(timesteps):
            # Expand timestep to batch dimension
            timestep_batch = t.expand(shape[0])
            
            # Predict noise
            if conditioning is not None:
                noise_pred = model_to_sample(sample, timestep_batch, conditioning)
            else:
                noise_pred = model_to_sample(sample, timestep_batch)
            
            # Apply classifier-free guidance if enabled
            if guidance_scale > 1.0 and conditioning is not None:
                # Unconditional prediction
                noise_pred_uncond = model_to_sample(sample, timestep_batch)
                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # Compute previous sample
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.noise_scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0, device=self.device)
            )
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            if self.prediction_type == "epsilon":
                # Predict x_0
                pred_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            elif self.prediction_type == "sample":
                pred_original_sample = noise_pred
            else:  # v_prediction
                pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * noise_pred
            
            # Compute coefficients for prev_sample
            pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            
            sample = prev_sample
        
        return sample
    
    @torch.no_grad()
    def sample_ddim(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        conditioning: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generate samples using DDIM sampling."""
        model_to_sample = self.ema_model if self.ema_model else self.model
        model_to_sample.eval()
        
        # Create random noise
        sample = torch.randn(shape, device=self.device, generator=generator)
        
        # Create timesteps for inference
        timesteps = torch.linspace(
            self.noise_scheduler.timesteps - 1, 0, num_inference_steps, device=self.device
        ).long()
        
        for i, t in enumerate(timesteps):
            timestep_batch = t.expand(shape[0])
            
            # Predict noise
            if conditioning is not None:
                noise_pred = model_to_sample(sample, timestep_batch, conditioning)
            else:
                noise_pred = model_to_sample(sample, timestep_batch)
            
            # Get alpha values
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.noise_scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0, device=self.device)
            )
            
            beta_prod_t = 1 - alpha_prod_t
            
            # Predict x_0
            if self.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            elif self.prediction_type == "sample":
                pred_original_sample = noise_pred
            else:  # v_prediction
                pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * noise_pred
            
            # Compute variance
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            std_dev_t = eta * variance ** 0.5
            
            # Compute direction pointing to x_t
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * noise_pred
            
            # Compute x_{t-1}
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            
            if eta > 0:
                noise = torch.randn_like(sample, generator=generator)
                prev_sample += std_dev_t * noise
            
            sample = prev_sample
        
        return sample
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save diffusion training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'noise_scheduler': self.noise_scheduler.__dict__,
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss,
            'prediction_type': self.prediction_type
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.ema_model:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Diffusion checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load diffusion training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.ema_model and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.prediction_type = checkpoint.get('prediction_type', 'epsilon')
        
        self.logger.info(f"Diffusion checkpoint loaded: {path}")
        return checkpoint


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpointManager:
    """Advanced checkpoint management with versioning and cleanup."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        
        self.checkpoints = []
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint = None
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load information about existing checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        for checkpoint_file in checkpoint_files:
            try:
                epoch = int(checkpoint_file.stem.split('_')[-1])
                self.checkpoints.append((epoch, checkpoint_file))
            except ValueError:
                continue
        
        self.checkpoints.sort(key=lambda x: x[0])
    
    def save_checkpoint(
        self,
        trainer: BaseTrainer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """Save checkpoint and manage cleanup."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        
        # Check if this is the best checkpoint
        is_best = False
        if self.save_best and self.monitor in metrics:
            score = metrics[self.monitor]
            if self.mode == 'min':
                is_best = score < self.best_score
            else:
                is_best = score > self.best_score
            
            if is_best:
                self.best_score = score
                self.best_checkpoint = checkpoint_path
        
        # Save the checkpoint
        trainer.save_checkpoint(str(checkpoint_path), epoch, metrics, is_best)
        
        # Add to tracking
        self.checkpoints.append((epoch, checkpoint_path))
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch and remove oldest
            self.checkpoints.sort(key=lambda x: x[0])
            to_remove = self.checkpoints[:-self.max_checkpoints]
            
            for epoch, checkpoint_path in to_remove:
                if checkpoint_path != self.best_checkpoint and checkpoint_path.exists():
                    checkpoint_path.unlink()
            
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        if not self.checkpoints:
            return None
        
        latest_epoch, latest_path = max(self.checkpoints, key=lambda x: x[0])
        return str(latest_path) if latest_path.exists() else None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        if self.best_checkpoint and self.best_checkpoint.exists():
            return str(self.best_checkpoint)
        return None


class TrainingMonitor:
    """Monitor training progress and provide insights."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = {}
        self.start_time = None
        self.epoch_times = []
    
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """Log metrics for an epoch."""
        self.epoch_times.append(epoch_time)
        
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.metrics_history:
            return {}
        
        stats = {}
        
        # Time statistics
        if self.start_time:
            stats['total_training_time'] = time.time() - self.start_time
        
        if self.epoch_times:
            stats['avg_epoch_time'] = np.mean(self.epoch_times)
            stats['total_epochs'] = len(self.epoch_times)
            stats['estimated_time_remaining'] = 0  # Can be calculated if total epochs known
        
        # Metrics statistics
        for metric_name, values in self.metrics_history.items():
            if len(values) > 0:
                recent_values = values[-self.window_size:]
                stats[f'{metric_name}_current'] = values[-1]
                stats[f'{metric_name}_best'] = min(values) if 'loss' in metric_name else max(values)
                stats[f'{metric_name}_avg_recent'] = np.mean(recent_values)
                stats[f'{metric_name}_trend'] = self._calculate_trend(recent_values)
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def should_stop_training(self, current_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if training should be stopped based on various criteria."""
        # Check for NaN or infinite losses
        for key, value in current_metrics.items():
            if 'loss' in key and (math.isnan(value) or math.isinf(value)):
                return True, f"Training stopped due to {key} becoming {value}"
        
        # Check for loss explosion
        if 'train_loss' in self.metrics_history and len(self.metrics_history['train_loss']) > 10:
            recent_losses = self.metrics_history['train_loss'][-10:]
            if any(loss > 100 for loss in recent_losses):
                return True, "Training stopped due to loss explosion"
        
        return False, ""


class HyperparameterTuner:
    """Automatic hyperparameter tuning utilities."""
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]]):
        self.param_ranges = param_ranges
        self.trial_history = []
    
    def suggest_parameters(self, method: str = "random") -> Dict[str, float]:
        """Suggest hyperparameters based on the specified method."""
        if method == "random":
            return self._random_search()
        elif method == "grid":
            return self._grid_search()
        else:
            raise ValueError(f"Unknown tuning method: {method}")
    
    def _random_search(self) -> Dict[str, float]:
        """Random hyperparameter search."""
        params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if param_name in ['batch_size', 'epochs']:
                # Integer parameters
                params[param_name] = int(np.random.uniform(min_val, max_val))
            else:
                # Float parameters (log scale for learning rate)
                if 'learning_rate' in param_name:
                    params[param_name] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def _grid_search(self) -> Dict[str, float]:
        """Grid search implementation (simplified)."""
        # This is a simplified version - a full implementation would
        # systematically explore the parameter space
        return self._random_search()
    
    def record_trial(self, params: Dict[str, float], score: float):
        """Record the results of a hyperparameter trial."""
        self.trial_history.append({
            'params': params.copy(),
            'score': score,
            'timestamp': time.time()
        })
    
    def get_best_parameters(self) -> Optional[Dict[str, float]]:
        """Get the best parameters found so far."""
        if not self.trial_history:
            return None
        
        best_trial = min(self.trial_history, key=lambda x: x['score'])
        return best_trial['params']


class DistributedTrainingManager:
    """Manager for distributed training setup and coordination."""
    
    def __init__(self, backend: str = 'nccl'):
        self.backend = backend
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
    
    def setup(self, rank: int, world_size: int, master_addr: str = 'localhost', master_port: str = '12355'):
        """Setup distributed training environment."""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)
        
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()
        self.is_initialized = True
        
        # Set device for this process
        torch.cuda.set_device(self.local_rank)
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        if not self.is_initialized:
            return model
        
        return DDP(model, device_ids=[self.local_rank])
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across processes."""
        if self.is_initialized:
            dist.all_reduce(tensor, op)
        return tensor
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather metrics from all processes."""
        if not self.is_initialized:
            return metrics
        
        gathered_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=f'cuda:{self.local_rank}')
            self.all_reduce(tensor)
            gathered_metrics[key] = tensor.item() / self.world_size
        
        return gathered_metrics


class TrainingOrchestrator:
    """High-level orchestrator for training workflows."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.distributed_manager = DistributedTrainingManager() if config.distributed else None
        self.checkpoint_manager = ModelCheckpointManager(config.checkpoint_dir)
        self.training_monitor = TrainingMonitor()
        self.early_stopping = EarlyStopping(patience=getattr(config, 'early_stopping_patience', 10))
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_gan(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        gen_optimizer: Optional[torch.optim.Optimizer] = None,
        disc_optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Orchestrate GAN training."""
        # Setup optimizers if not provided
        if gen_optimizer is None:
            gen_optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2)
            )
        
        if disc_optimizer is None:
            disc_optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2)
            )
        
        # Create trainer
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=self.config,
            device=f'cuda:{self.config.rank}' if self.config.distributed else 'cuda'
        )
        
        # Setup distributed training
        if self.distributed_manager:
            self.distributed_manager.setup(self.config.rank, self.config.world_size)
            generator = self.distributed_manager.wrap_model(generator)
            discriminator = self.distributed_manager.wrap_model(discriminator)
        
        return self._run_training_loop(trainer, train_dataloader, val_dataloader)
    
    def train_diffusion(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        noise_scheduler: Optional[NoiseScheduler] = None
    ) -> Dict[str, Any]:
        """Orchestrate diffusion model training."""
        # Setup optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Setup noise scheduler if not provided
        if noise_scheduler is None:
            noise_scheduler = NoiseScheduler(
                schedule_type="cosine",
                timesteps=1000,
                device=f'cuda:{self.config.rank}' if self.config.distributed else 'cuda'
            )
        
        # Create trainer
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=self.config,
            noise_scheduler=noise_scheduler,
            device=f'cuda:{self.config.rank}' if self.config.distributed else 'cuda'
        )
        
        # Setup distributed training
        if self.distributed_manager:
            self.distributed_manager.setup(self.config.rank, self.config.world_size)
            model = self.distributed_manager.wrap_model(model)
        
        return self._run_training_loop(trainer, train_dataloader, val_dataloader)
    
    def _run_training_loop(
        self,
        trainer: BaseTrainer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Run the main training loop."""
        self.training_monitor.start_training()
        
        # Try to resume from checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            trainer.load_checkpoint(latest_checkpoint)
            self.logger.info(f"Resumed training from epoch {trainer.current_epoch}")
        
        training_results = {
            'best_epoch': 0,
            'best_metrics': {},
            'final_metrics': {},
            'training_stats': {},
            'stopped_early': False,
            'stop_reason': ''
        }
        
        try:
            for epoch in range(trainer.current_epoch, self.config.epochs):
                epoch_start_time = time.time()
                trainer.current_epoch = epoch
                
                # Training phase
                train_metrics = trainer.train_epoch(train_dataloader)
                
                # Validation phase
                val_metrics = {}
                if val_dataloader is not None:
                    val_metrics = trainer.validate(val_dataloader)
                
                # Combine metrics
                all_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                
                # Distributed metrics gathering
                if self.distributed_manager:
                    all_metrics = self.distributed_manager.gather_metrics(all_metrics)
                
                epoch_time = time.time() - epoch_start_time
                self.training_monitor.log_epoch(epoch, all_metrics, epoch_time)
                
                # Log metrics
                trainer.log_metrics(all_metrics, epoch, "epoch")
                
                # Check for training issues
                should_stop, stop_reason = self.training_monitor.should_stop_training(all_metrics)
                if should_stop:
                    training_results['stopped_early'] = True
                    training_results['stop_reason'] = stop_reason
                    self.logger.warning(f"Training stopped: {stop_reason}")
                    break
                
                # Early stopping check
                val_loss = val_metrics.get('diffusion_loss', val_metrics.get('gen_loss', float('inf')))
                if self.early_stopping(val_loss):
                    training_results['stopped_early'] = True
                    training_results['stop_reason'] = "Early stopping triggered"
                    self.logger.info("Early stopping triggered")
                    break
                
                # Save checkpoint
                if (epoch + 1) % self.config.checkpoint_freq == 0:
                    self.checkpoint_manager.save_checkpoint(trainer, epoch, all_metrics)
                
                # Update best metrics
                if val_loss < training_results.get('best_val_loss', float('inf')):
                    training_results['best_epoch'] = epoch
                    training_results['best_metrics'] = all_metrics.copy()
                    training_results['best_val_loss'] = val_loss
                
                # Log progress
                if self.distributed_manager is None or self.distributed_manager.is_main_process():
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} - "
                        f"Train Loss: {train_metrics.get('diffusion_loss', train_metrics.get('gen_loss', 0)):.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Time: {epoch_time:.2f}s"
                    )
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            training_results['stopped_early'] = True
            training_results['stop_reason'] = "User interruption"
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            training_results['stopped_early'] = True
            training_results['stop_reason'] = f"Error: {str(e)}"
            raise
        
        finally:
            # Cleanup
            if self.distributed_manager:
                self.distributed_manager.cleanup()
        
        # Final results
        training_results['final_metrics'] = all_metrics
        training_results['training_stats'] = self.training_monitor.get_training_stats()
        
        return training_results


# Utility functions for creating optimizers and schedulers
def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 0.0002,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer for model parameters."""
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=kwargs.get('betas', (0.5, 0.999)),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Example usage and factory functions
def create_training_config(**kwargs) -> TrainingConfig:
    """Create training configuration with sensible defaults."""
    return TrainingConfig(**kwargs)


def setup_training_environment(config: TrainingConfig) -> Dict[str, Any]:
    """Setup complete training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    if torch.cuda.is_available():
        device = f'cuda:{config.rank}' if config.distributed else 'cuda'
        torch.cuda.manual_seed(42)
    else:
        device = 'cpu'
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.snapshot_dir, exist_ok=True)
    
    return {
        'device': device,
        'checkpoint_dir': config.checkpoint_dir,
        'log_dir': config.log_dir,
        'snapshot_dir': config.snapshot_dir
    }