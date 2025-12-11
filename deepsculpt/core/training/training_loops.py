"""
Training loops for DeepSculpt PyTorch implementation.

This module provides specialized training loops for different model types,
handling the core training logic and optimization steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # New PyTorch 2.9+ API
    from torch.amp import autocast, GradScaler
    def get_autocast(device='cuda'):
        return autocast(device if device != 'cpu' else 'cpu')
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler
    def get_autocast(device='cuda'):
        return autocast()
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np


class GANTrainingLoop:
    """
    Training loop for GAN models with various loss functions and training strategies.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        loss_type: str = "bce",
        use_gradient_penalty: bool = False,
        gradient_penalty_weight: float = 10.0,
        device: str = "cuda",
        gen_train_freq: int = 1,
        disc_train_freq: int = 1
    ):
        """
        Initialize GAN training loop.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            loss_type: Type of adversarial loss ("bce", "wgan", "lsgan", "hinge")
            use_gradient_penalty: Whether to use gradient penalty
            gradient_penalty_weight: Weight for gradient penalty
            device: Device for training
            gen_train_freq: Generator training frequency
            disc_train_freq: Discriminator training frequency
        """
        self.generator = generator
        self.discriminator = discriminator
        self.loss_type = loss_type
        self.use_gradient_penalty = use_gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight
        self.device = device
        self.gen_train_freq = gen_train_freq
        self.disc_train_freq = disc_train_freq
        
        self.step_count = 0
    
    def adversarial_loss(self, output: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate adversarial loss based on loss type."""
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
    
    def train_step(
        self,
        real_data: torch.Tensor,
        noise: torch.Tensor,
        gen_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        gen_scaler: Optional[GradScaler] = None,
        disc_scaler: Optional[GradScaler] = None,
        mixed_precision: bool = False,
        gradient_clip: float = 0.0
    ) -> Dict[str, float]:
        """
        Execute a single GAN training step.
        
        Args:
            real_data: Batch of real data
            noise: Noise for generator
            gen_optimizer: Generator optimizer
            disc_optimizer: Discriminator optimizer
            gen_scaler: Generator gradient scaler for mixed precision
            disc_scaler: Discriminator gradient scaler for mixed precision
            mixed_precision: Whether to use mixed precision
            gradient_clip: Gradient clipping value
            
        Returns:
            Dictionary of step metrics
        """
        batch_size = real_data.size(0)
        metrics = {}
        
        # Train Discriminator
        if self.step_count % self.disc_train_freq == 0:
            disc_optimizer.zero_grad()
            
            if mixed_precision and disc_scaler:
                with get_autocast():
                    disc_metrics = self._train_discriminator_step(real_data, noise)
                
                disc_scaler.scale(disc_metrics['disc_loss']).backward()
                
                if gradient_clip > 0:
                    disc_scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), gradient_clip)
                
                disc_scaler.step(disc_optimizer)
                disc_scaler.update()
            else:
                disc_metrics = self._train_discriminator_step(real_data, noise)
                disc_metrics['disc_loss'].backward()
                
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), gradient_clip)
                
                disc_optimizer.step()
            
            metrics.update(disc_metrics)
        
        # Train Generator
        if self.step_count % self.gen_train_freq == 0:
            gen_optimizer.zero_grad()
            
            if mixed_precision and gen_scaler:
                with get_autocast():
                    gen_metrics = self._train_generator_step(noise)
                
                gen_scaler.scale(gen_metrics['gen_loss']).backward()
                
                if gradient_clip > 0:
                    gen_scaler.unscale_(gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), gradient_clip)
                
                gen_scaler.step(gen_optimizer)
                gen_scaler.update()
            else:
                gen_metrics = self._train_generator_step(noise)
                gen_metrics['gen_loss'].backward()
                
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), gradient_clip)
                
                gen_optimizer.step()
            
            metrics.update(gen_metrics)
        
        self.step_count += 1
        
        # Convert tensors to floats
        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
    
    def _train_discriminator_step(self, real_data: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Train discriminator for one step."""
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
        gp_loss = torch.tensor(0.0, device=self.device)
        if self.use_gradient_penalty:
            gp_loss = self.gradient_penalty(real_data, fake_data)
            disc_loss += self.gradient_penalty_weight * gp_loss
        
        # Calculate accuracies
        with torch.no_grad():
            if self.loss_type == "bce":
                real_acc = (torch.sigmoid(real_output) > 0.5).float().mean()
                fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean()
            else:
                # For other loss types, use sign-based accuracy
                real_acc = (real_output > 0).float().mean()
                fake_acc = (fake_output < 0).float().mean()
        
        return {
            'disc_loss': disc_loss,
            'disc_real_loss': real_loss,
            'disc_fake_loss': fake_loss,
            'disc_real_acc': real_acc,
            'disc_fake_acc': fake_acc,
            'gradient_penalty': gp_loss
        }
    
    def _train_generator_step(self, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Train generator for one step."""
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        gen_loss = self.adversarial_loss(fake_output, True)
        
        return {
            'gen_loss': gen_loss
        }


class DiffusionTrainingLoop:
    """
    Training loop for diffusion models with various prediction types.
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler,
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
        conditioning_dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        Initialize diffusion training loop.
        
        Args:
            model: Diffusion model
            noise_scheduler: Noise scheduler
            prediction_type: Type of prediction ("epsilon", "sample", "v_prediction")
            loss_type: Type of loss ("mse", "l1", "huber")
            conditioning_dropout: Dropout rate for classifier-free guidance
            device: Device for training
        """
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.conditioning_dropout = conditioning_dropout
        self.device = device
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
        mixed_precision: bool = False,
        gradient_clip: float = 0.0,
        conditioning_key: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Execute a single diffusion training step.
        
        Args:
            batch: Batch of training data
            optimizer: Model optimizer
            scaler: Gradient scaler for mixed precision
            mixed_precision: Whether to use mixed precision
            gradient_clip: Gradient clipping value
            conditioning_key: Key for conditioning information
            
        Returns:
            Dictionary of step metrics
        """
        # Extract data from batch
        if isinstance(batch, dict):
            x_0 = batch.get('data', batch.get('structure', batch.get('x', None)))
            conditioning = batch.get(conditioning_key) if conditioning_key else None
        else:
            x_0 = batch
            conditioning = None
        
        if x_0 is None:
            raise ValueError("Could not find data in batch")
        
        x_0 = x_0.to(self.device)
        if conditioning is not None:
            conditioning = conditioning.to(self.device)
        
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.timesteps, (batch_size,), device=self.device, dtype=torch.long
        )
        
        # Add noise to samples
        noise = torch.randn_like(x_0)
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)
        
        # Apply conditioning dropout for classifier-free guidance
        if conditioning is not None and self.conditioning_dropout > 0:
            dropout_mask = torch.rand(batch_size, device=self.device) < self.conditioning_dropout
            conditioning = conditioning.clone()
            conditioning[dropout_mask] = 0  # Zero out conditioning for dropped samples
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler:
            with get_autocast():
                metrics = self._forward_pass(x_t, timesteps, x_0, noise, conditioning)
                loss = metrics['diffusion_loss']
            
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            metrics = self._forward_pass(x_t, timesteps, x_0, noise, conditioning)
            loss = metrics['diffusion_loss']
            
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()
        
        # Convert tensors to floats
        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
    
    def _forward_pass(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass and loss computation."""
        # Get model prediction
        if conditioning is not None:
            model_output = self.model(x_t, timesteps, conditioning)
        else:
            model_output = self.model(x_t, timesteps)
        
        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x_0
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x_0, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Compute losses
        if self.loss_type == "mse":
            main_loss = F.mse_loss(model_output, target)
        elif self.loss_type == "l1":
            main_loss = F.l1_loss(model_output, target)
        elif self.loss_type == "huber":
            main_loss = F.huber_loss(model_output, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {
            'diffusion_loss': main_loss,
            'mse_loss': F.mse_loss(model_output, target),
            'l1_loss': F.l1_loss(model_output, target)
        }


class ProgressiveTrainingLoop:
    """
    Training loop for progressive growing models.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        base_training_loop: GANTrainingLoop,
        growth_schedule: List[int],
        fade_in_steps: int = 1000
    ):
        """
        Initialize progressive training loop.
        
        Args:
            generator: Progressive generator model
            discriminator: Progressive discriminator model
            base_training_loop: Base GAN training loop
            growth_schedule: List of steps at which to grow the network
            fade_in_steps: Number of steps for fade-in transition
        """
        self.generator = generator
        self.discriminator = discriminator
        self.base_training_loop = base_training_loop
        self.growth_schedule = growth_schedule
        self.fade_in_steps = fade_in_steps
        
        self.current_level = 0
        self.steps_since_growth = 0
        self.total_steps = 0
    
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """Execute a progressive training step."""
        # Check if we need to grow the network
        if self.total_steps in self.growth_schedule and self.current_level < len(self.growth_schedule) - 1:
            self._grow_network()
        
        # Update alpha for fade-in
        if self.steps_since_growth < self.fade_in_steps:
            alpha = self.steps_since_growth / self.fade_in_steps
            if hasattr(self.generator, 'set_alpha'):
                self.generator.set_alpha(alpha)
            if hasattr(self.discriminator, 'set_alpha'):
                self.discriminator.set_alpha(alpha)
        
        # Execute base training step
        metrics = self.base_training_loop.train_step(*args, **kwargs)
        
        # Add progressive-specific metrics
        metrics['progressive_level'] = self.current_level
        metrics['progressive_alpha'] = getattr(self.generator, 'alpha', 1.0)
        
        self.steps_since_growth += 1
        self.total_steps += 1
        
        return metrics
    
    def _grow_network(self):
        """Grow the network to the next level."""
        if hasattr(self.generator, 'grow'):
            self.generator.grow()
        if hasattr(self.discriminator, 'grow'):
            self.discriminator.grow()
        
        self.current_level += 1
        self.steps_since_growth = 0
        
        print(f"Progressive growing: advanced to level {self.current_level}")


class CurriculumTrainingLoop:
    """
    Training loop with curriculum learning.
    """
    
    def __init__(
        self,
        base_training_loop: Union[GANTrainingLoop, DiffusionTrainingLoop],
        curriculum_stages: List[Dict[str, Any]],
        advancement_criteria: Dict[str, float]
    ):
        """
        Initialize curriculum training loop.
        
        Args:
            base_training_loop: Base training loop
            curriculum_stages: List of curriculum stage configurations
            advancement_criteria: Criteria for advancing to next stage
        """
        self.base_training_loop = base_training_loop
        self.curriculum_stages = curriculum_stages
        self.advancement_criteria = advancement_criteria
        
        self.current_stage = 0
        self.stage_metrics_history = []
    
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """Execute a curriculum training step."""
        # Apply current curriculum stage settings
        self._apply_curriculum_stage()
        
        # Execute base training step
        metrics = self.base_training_loop.train_step(*args, **kwargs)
        
        # Track metrics for curriculum advancement
        self.stage_metrics_history.append(metrics)
        
        # Check if we should advance to next stage
        if self._should_advance_stage():
            self._advance_stage()
        
        # Add curriculum-specific metrics
        metrics['curriculum_stage'] = self.current_stage
        
        return metrics
    
    def _apply_curriculum_stage(self):
        """Apply settings for current curriculum stage."""
        if self.current_stage < len(self.curriculum_stages):
            stage_config = self.curriculum_stages[self.current_stage]
            
            # Apply stage-specific configurations
            for key, value in stage_config.items():
                if hasattr(self.base_training_loop, key):
                    setattr(self.base_training_loop, key, value)
    
    def _should_advance_stage(self) -> bool:
        """Check if we should advance to the next curriculum stage."""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False
        
        if len(self.stage_metrics_history) < 100:  # Need enough history
            return False
        
        # Check advancement criteria
        recent_metrics = self.stage_metrics_history[-50:]  # Last 50 steps
        
        for metric_name, threshold in self.advancement_criteria.items():
            if metric_name in recent_metrics[0]:  # Check if metric exists
                avg_metric = np.mean([m[metric_name] for m in recent_metrics])
                if avg_metric < threshold:  # Assuming lower is better
                    return False
        
        return True
    
    def _advance_stage(self):
        """Advance to the next curriculum stage."""
        self.current_stage += 1
        self.stage_metrics_history = []  # Reset history for new stage
        print(f"Curriculum learning: advanced to stage {self.current_stage}")


# Convenience functions for creating training loops
def create_gan_training_loop(
    generator: nn.Module,
    discriminator: nn.Module,
    loss_type: str = "bce",
    **kwargs
) -> GANTrainingLoop:
    """Create a GAN training loop with default settings."""
    return GANTrainingLoop(generator, discriminator, loss_type, **kwargs)


def create_diffusion_training_loop(
    model: nn.Module,
    noise_scheduler,
    prediction_type: str = "epsilon",
    **kwargs
) -> DiffusionTrainingLoop:
    """Create a diffusion training loop with default settings."""
    return DiffusionTrainingLoop(model, noise_scheduler, prediction_type, **kwargs)