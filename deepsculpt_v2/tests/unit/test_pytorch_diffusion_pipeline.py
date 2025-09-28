"""
Comprehensive tests for PyTorch diffusion pipeline implementation.

This module tests all components of the diffusion pipeline including:
- Noise scheduling algorithms
- Advanced sampling techniques
- Forward and reverse diffusion processes
- Pipeline integration and correctness
- Performance and memory efficiency
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional

# Import the diffusion components to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deepSculpt.pytorch_diffusion import (
    NoiseScheduler,
    NoiseScheduleConfig,
    ScheduleType,
    NoiseType,
    AdvancedSampler,
    SamplingConfig,
    SamplingMethod,
    Diffusion3DPipeline
)

# Import UNet model for testing
from deepSculpt.pytorch_models import UNet3D


class SimpleTestModel(nn.Module):
    """Simple model for testing diffusion pipeline."""
    
    def __init__(self, channels: int = 4):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv3d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, channels, 3, padding=1)
        self.time_embed = nn.Linear(256, 64)
        
    def forward(self, x, timesteps, conditioning=None):
        # Simple time embedding
        t_emb = self.time_embed(self._get_time_embedding(timesteps))
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1, 1)
        
        # Simple forward pass
        h = torch.relu(self.conv1(x))
        h = h + t_emb  # Add time embedding
        h = torch.relu(self.conv2(h))
        return self.conv3(h)
    
    def _get_time_embedding(self, timesteps):
        """Simple sinusoidal time embedding."""
        half_dim = 128
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class TestNoiseScheduler:
    """Test noise scheduling algorithms."""
    
    def test_linear_schedule(self):
        """Test linear noise schedule."""
        config = NoiseScheduleConfig(
            schedule_type=ScheduleType.LINEAR,
            timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02
        )
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Check schedule properties
        assert scheduler.betas.shape == (1000,)
        assert torch.allclose(scheduler.betas[0], torch.tensor(0.0001), atol=1e-6)
        assert torch.allclose(scheduler.betas[-1], torch.tensor(0.02), atol=1e-6)
        
        # Check monotonicity
        assert torch.all(scheduler.betas[1:] >= scheduler.betas[:-1])
    
    def test_cosine_schedule(self):
        """Test cosine noise schedule."""
        config = NoiseScheduleConfig(
            schedule_type=ScheduleType.COSINE,
            timesteps=1000,
            cosine_s=0.008
        )
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Check schedule properties
        assert scheduler.betas.shape == (1000,)
        assert torch.all(scheduler.betas > 0)
        assert torch.all(scheduler.betas < 1)
        
        # Cosine schedule should start small and increase
        assert scheduler.betas[0] < scheduler.betas[500]
    
    def test_sigmoid_schedule(self):
        """Test sigmoid noise schedule."""
        config = NoiseScheduleConfig(
            schedule_type=ScheduleType.SIGMOID,
            timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            sigmoid_start=-3.0,
            sigmoid_end=3.0
        )
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Check schedule properties
        assert scheduler.betas.shape == (1000,)
        assert torch.all(scheduler.betas >= 0.0001)
        assert torch.all(scheduler.betas <= 0.02)
    
    def test_exponential_schedule(self):
        """Test exponential noise schedule."""
        config = NoiseScheduleConfig(
            schedule_type=ScheduleType.EXPONENTIAL,
            timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            exp_gamma=0.9
        )
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Check schedule properties
        assert scheduler.betas.shape == (1000,)
        assert torch.all(scheduler.betas >= 0.0001)
        assert torch.all(scheduler.betas <= 0.02)
    
    def test_custom_schedule(self):
        """Test custom noise schedule."""
        custom_betas = torch.linspace(0.0001, 0.02, 1000)
        config = NoiseScheduleConfig(
            schedule_type=ScheduleType.CUSTOM,
            timesteps=1000,
            custom_betas=custom_betas
        )
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Check that custom betas are used
        torch.testing.assert_close(scheduler.betas, custom_betas)
    
    def test_precomputed_values(self):
        """Test that precomputed values are correct."""
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Check alphas
        expected_alphas = 1.0 - scheduler.betas
        torch.testing.assert_close(scheduler.alphas, expected_alphas)
        
        # Check alphas_cumprod
        expected_alphas_cumprod = torch.cumprod(expected_alphas, dim=0)
        torch.testing.assert_close(scheduler.alphas_cumprod, expected_alphas_cumprod)
        
        # Check sqrt values
        torch.testing.assert_close(
            scheduler.sqrt_alphas_cumprod,
            torch.sqrt(expected_alphas_cumprod)
        )
    
    def test_noise_sampling(self):
        """Test different noise sampling methods."""
        config = NoiseScheduleConfig()
        scheduler = NoiseScheduler(config, device="cpu")
        
        shape = (2, 4, 16, 16, 16)
        
        # Test Gaussian noise
        gaussian_noise = scheduler.sample_noise(shape, NoiseType.GAUSSIAN)
        assert gaussian_noise.shape == shape
        assert torch.allclose(gaussian_noise.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(gaussian_noise.std(), torch.tensor(1.0), atol=0.1)
        
        # Test uniform noise
        uniform_noise = scheduler.sample_noise(shape, NoiseType.UNIFORM)
        assert uniform_noise.shape == shape
        assert torch.all(uniform_noise >= -1)
        assert torch.all(uniform_noise <= 1)
    
    def test_add_noise(self):
        """Test noise addition process."""
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Create test data
        batch_size = 2
        original_samples = torch.randn(batch_size, 4, 8, 8, 8)
        timesteps = torch.randint(0, 100, (batch_size,))
        
        # Add noise
        noisy_samples, noise = scheduler.add_noise(original_samples, timesteps=timesteps)
        
        # Check shapes
        assert noisy_samples.shape == original_samples.shape
        assert noise.shape == original_samples.shape
        
        # Check that noise was actually added
        assert not torch.allclose(noisy_samples, original_samples)
    
    def test_velocity_parameterization(self):
        """Test velocity parameterization."""
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        sample = torch.randn(1, 4, 8, 8, 8)
        noise = torch.randn(1, 4, 8, 8, 8)
        timesteps = torch.randint(0, 100, (1,))
        
        velocity = scheduler.get_velocity(sample, noise, timesteps)
        
        assert velocity.shape == sample.shape
        assert not torch.allclose(velocity, sample)
        assert not torch.allclose(velocity, noise)
    
    def test_predict_start_from_noise(self):
        """Test prediction of x_0 from noise."""
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Create test scenario
        x_0 = torch.randn(1, 4, 8, 8, 8)
        timesteps = torch.tensor([50])
        
        # Add noise to get x_t
        x_t, noise = scheduler.add_noise(x_0, timesteps=timesteps)
        
        # Predict x_0 from x_t and noise
        pred_x_0 = scheduler.predict_start_from_noise(x_t, timesteps, noise)
        
        # Should be close to original (not exact due to numerical precision)
        assert pred_x_0.shape == x_0.shape
        assert torch.allclose(pred_x_0, x_0, atol=1e-4)
    
    def test_adaptive_scheduling(self):
        """Test adaptive noise scheduling."""
        config = NoiseScheduleConfig(
            timesteps=100,
            adaptive=True,
            adaptation_rate=0.1,
            target_snr=0.5
        )
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Store original betas
        original_betas = scheduler.betas.clone()
        
        # Simulate high loss (should reduce noise)
        high_loss_metrics = {'loss': 1.0}
        for _ in range(15):  # Need enough history
            scheduler.adapt_schedule(high_loss_metrics)
        
        # Betas should have changed
        assert not torch.allclose(scheduler.betas, original_betas)
    
    def test_schedule_statistics(self):
        """Test schedule statistics tracking."""
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Generate some samples to create statistics
        for _ in range(10):
            samples = torch.randn(2, 4, 8, 8, 8)
            scheduler.add_noise(samples)
        
        stats = scheduler.get_schedule_stats()
        
        assert 'schedule_type' in stats
        assert 'timesteps' in stats
        assert 'total_samples' in stats
        assert stats['total_samples'] > 0


class TestAdvancedSampler:
    """Test advanced sampling algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = NoiseScheduleConfig(timesteps=100)
        self.scheduler = NoiseScheduler(self.config, device="cpu")
        self.model = SimpleTestModel()
        self.model.eval()
    
    def test_ddpm_sampling(self):
        """Test DDPM sampling."""
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10
        )
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        samples = sampler.sample(self.model, shape)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
    
    def test_ddim_sampling(self):
        """Test DDIM sampling."""
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDIM,
            num_inference_steps=10,
            eta=0.0
        )
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        samples = sampler.sample(self.model, shape)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
    
    def test_ddim_with_eta(self):
        """Test DDIM sampling with eta > 0."""
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDIM,
            num_inference_steps=10,
            eta=0.5
        )
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        samples = sampler.sample(self.model, shape)
        
        assert samples.shape == shape
    
    def test_conditional_sampling(self):
        """Test sampling with conditioning."""
        sampling_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=5)
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        conditioning = torch.randn(1, 64)  # Some conditioning vector
        
        samples = sampler.sample(self.model, shape, conditioning=conditioning)
        
        assert samples.shape == shape
    
    def test_img2img_sampling(self):
        """Test image-to-image sampling."""
        sampling_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=5)
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        init_image = torch.randn(1, 4, 8, 8, 8)
        strength = 0.7
        
        samples = sampler.sample(self.model, init_image.shape, init_image=init_image, strength=strength)
        
        assert samples.shape == init_image.shape
        # Should be different from init_image due to transformation
        assert not torch.allclose(samples, init_image, atol=0.1)
    
    def test_inpainting_sampling(self):
        """Test inpainting with mask."""
        sampling_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=5)
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        init_image = torch.randn(shape)
        mask = torch.zeros(shape)
        mask[:, :, 2:6, 2:6, 2:6] = 1  # Mask center region
        
        samples = sampler.sample(self.model, shape, mask=mask, init_image=init_image)
        
        assert samples.shape == shape
    
    def test_classifier_free_guidance(self):
        """Test classifier-free guidance."""
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDIM,
            num_inference_steps=5,
            guidance_scale=2.0
        )
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        conditioning = torch.randn(1, 64)
        
        samples = sampler.sample(self.model, shape, conditioning=conditioning)
        
        assert samples.shape == shape
    
    def test_sampling_callback(self):
        """Test sampling with progress callback."""
        sampling_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=5)
        sampler = AdvancedSampler(self.scheduler, sampling_config)
        
        callback_calls = []
        
        def progress_callback(step, total_steps, current_sample):
            callback_calls.append((step, total_steps, current_sample.shape))
        
        shape = (1, 4, 8, 8, 8)
        samples = sampler.sample(self.model, shape, callback=progress_callback)
        
        assert len(callback_calls) > 0
        assert all(call[2] == shape for call in callback_calls)


class TestDiffusion3DPipeline:
    """Test complete diffusion pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = NoiseScheduleConfig(timesteps=100)
        self.scheduler = NoiseScheduler(self.config, device="cpu")
        self.model = SimpleTestModel()
        self.pipeline = Diffusion3DPipeline(self.model, self.scheduler, device="cpu")
    
    def test_forward_process(self):
        """Test forward diffusion process."""
        x_0 = torch.randn(2, 4, 8, 8, 8)
        
        x_t, noise, timesteps = self.pipeline.forward_process(x_0)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
        assert timesteps.shape == (2,)
        assert not torch.allclose(x_t, x_0)  # Should be noisy
    
    def test_forward_process_with_timesteps(self):
        """Test forward process with specific timesteps."""
        x_0 = torch.randn(2, 4, 8, 8, 8)
        t = torch.tensor([10, 50])
        
        x_t, noise, returned_t = self.pipeline.forward_process(x_0, t=t)
        
        assert torch.equal(returned_t, t)
        assert x_t.shape == x_0.shape
    
    def test_reverse_process(self):
        """Test reverse diffusion process."""
        x_t = torch.randn(1, 4, 8, 8, 8)
        t = torch.tensor([50])
        
        x_prev = self.pipeline.reverse_process(x_t, t)
        
        assert x_prev.shape == x_t.shape
        assert not torch.allclose(x_prev, x_t)  # Should be different
    
    def test_sample_generation(self):
        """Test sample generation."""
        shape = (1, 4, 8, 8, 8)
        
        samples = self.pipeline.sample(shape)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
    
    def test_conditional_sample_generation(self):
        """Test conditional sample generation."""
        shape = (1, 4, 8, 8, 8)
        conditioning = torch.randn(1, 64)
        
        samples = self.pipeline.sample(shape, conditioning=conditioning)
        
        assert samples.shape == shape
    
    def test_loss_computation(self):
        """Test training loss computation."""
        x_0 = torch.randn(2, 4, 8, 8, 8)
        
        losses = self.pipeline.compute_loss(x_0)
        
        assert 'mse_loss' in losses
        assert 'l1_loss' in losses
        assert 'combined_loss' in losses
        
        # All losses should be positive
        for loss_name, loss_value in losses.items():
            assert loss_value.item() >= 0
    
    def test_conditional_loss_computation(self):
        """Test conditional training loss computation."""
        x_0 = torch.randn(2, 4, 8, 8, 8)
        conditioning = torch.randn(2, 64)
        
        losses = self.pipeline.compute_loss(x_0, conditioning=conditioning)
        
        assert 'mse_loss' in losses
        assert losses['mse_loss'].item() >= 0
    
    def test_different_loss_types(self):
        """Test different loss types."""
        x_0 = torch.randn(1, 4, 8, 8, 8)
        
        # Test MSE loss
        mse_losses = self.pipeline.compute_loss(x_0, loss_type="mse")
        assert 'mse_loss' in mse_losses
        
        # Test L1 loss
        l1_losses = self.pipeline.compute_loss(x_0, loss_type="l1")
        assert 'l1_loss' in l1_losses
        
        # Test Huber loss
        huber_losses = self.pipeline.compute_loss(x_0, loss_type="huber")
        assert 'huber_loss' in huber_losses
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics tracking."""
        # Generate some samples and compute losses
        x_0 = torch.randn(2, 4, 8, 8, 8)
        self.pipeline.compute_loss(x_0)
        
        shape = (1, 4, 8, 8, 8)
        self.pipeline.sample(shape)
        
        stats = self.pipeline.get_pipeline_stats()
        
        assert 'total_training_steps' in stats
        assert 'total_samples_generated' in stats
        assert 'scheduler_stats' in stats
        assert 'model_parameters' in stats
        
        assert stats['total_training_steps'] > 0
        assert stats['total_samples_generated'] > 0
    
    def test_pipeline_save_load(self):
        """Test pipeline save and load functionality."""
        import tempfile
        
        # Train a bit to have some state
        x_0 = torch.randn(1, 4, 8, 8, 8)
        self.pipeline.compute_loss(x_0)
        
        # Save pipeline
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            self.pipeline.save_pipeline(f.name)
            
            # Create new pipeline and load
            new_pipeline = Diffusion3DPipeline(
                SimpleTestModel(), 
                NoiseScheduler(self.config, device="cpu"),
                device="cpu"
            )
            new_pipeline.load_pipeline(f.name)
            
            # Check that states match
            assert new_pipeline.stats['total_training_steps'] == self.pipeline.stats['total_training_steps']
            
            # Clean up
            os.unlink(f.name)


class TestDiffusionIntegration:
    """Integration tests for diffusion components."""
    
    def test_unet_integration(self):
        """Test integration with actual UNet model."""
        # Create UNet model
        unet = UNet3D(in_channels=4, out_channels=4, base_channels=32, num_levels=2)
        
        # Create pipeline
        config = NoiseScheduleConfig(timesteps=50)  # Smaller for faster testing
        scheduler = NoiseScheduler(config, device="cpu")
        pipeline = Diffusion3DPipeline(unet, scheduler, device="cpu")
        
        # Test forward and reverse processes
        x_0 = torch.randn(1, 4, 16, 16, 16)
        
        # Forward process
        x_t, noise, t = pipeline.forward_process(x_0)
        assert x_t.shape == x_0.shape
        
        # Reverse process
        x_prev = pipeline.reverse_process(x_t, t)
        assert x_prev.shape == x_0.shape
        
        # Sample generation
        samples = pipeline.sample((1, 4, 16, 16, 16))
        assert samples.shape == (1, 4, 16, 16, 16)
    
    def test_different_sampling_methods(self):
        """Test different sampling methods with same model."""
        model = SimpleTestModel()
        config = NoiseScheduleConfig(timesteps=20)
        scheduler = NoiseScheduler(config, device="cpu")
        
        shape = (1, 4, 8, 8, 8)
        
        # Test DDPM
        ddpm_config = SamplingConfig(method=SamplingMethod.DDPM, num_inference_steps=5)
        ddpm_sampler = AdvancedSampler(scheduler, ddpm_config)
        ddpm_samples = ddpm_sampler.sample(model, shape)
        
        # Test DDIM
        ddim_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=5)
        ddim_sampler = AdvancedSampler(scheduler, ddim_config)
        ddim_samples = ddim_sampler.sample(model, shape)
        
        # Both should produce valid samples
        assert ddpm_samples.shape == shape
        assert ddim_samples.shape == shape
        assert not torch.isnan(ddpm_samples).any()
        assert not torch.isnan(ddim_samples).any()
        
        # Samples should be different (due to different algorithms)
        assert not torch.allclose(ddpm_samples, ddim_samples, atol=0.1)
    
    def test_batch_processing(self):
        """Test pipeline with different batch sizes."""
        model = SimpleTestModel()
        config = NoiseScheduleConfig(timesteps=20)
        scheduler = NoiseScheduler(config, device="cpu")
        pipeline = Diffusion3DPipeline(model, scheduler, device="cpu")
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            shape = (batch_size, 4, 8, 8, 8)
            
            # Test forward process
            x_0 = torch.randn(shape)
            x_t, noise, t = pipeline.forward_process(x_0)
            assert x_t.shape == shape
            
            # Test loss computation
            losses = pipeline.compute_loss(x_0)
            assert losses['mse_loss'].item() >= 0
            
            # Test sampling
            samples = pipeline.sample(shape)
            assert samples.shape == shape
    
    def test_memory_efficiency(self):
        """Test memory usage during diffusion process."""
        model = SimpleTestModel()
        config = NoiseScheduleConfig(timesteps=10)
        scheduler = NoiseScheduler(config, device="cpu")
        pipeline = Diffusion3DPipeline(model, scheduler, device="cpu")
        
        # Test with larger samples
        shape = (2, 4, 32, 32, 32)
        
        # Should not run out of memory
        x_0 = torch.randn(shape)
        losses = pipeline.compute_loss(x_0)
        assert losses['mse_loss'].item() >= 0
        
        # Sampling should also work
        samples = pipeline.sample(shape)
        assert samples.shape == shape
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the pipeline."""
        model = SimpleTestModel()
        config = NoiseScheduleConfig(timesteps=10)
        scheduler = NoiseScheduler(config, device="cpu")
        pipeline = Diffusion3DPipeline(model, scheduler, device="cpu")
        
        # Enable gradients
        model.train()
        x_0 = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
        
        # Compute loss
        losses = pipeline.compute_loss(x_0)
        loss = losses['combined_loss']
        
        # Backward pass
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
        
        # Check that input has gradients
        assert x_0.grad is not None
    
    def test_deterministic_sampling(self):
        """Test that sampling can be made deterministic."""
        model = SimpleTestModel()
        config = NoiseScheduleConfig(timesteps=10)
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Set random seed
        torch.manual_seed(42)
        
        sampling_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=5)
        sampler = AdvancedSampler(scheduler, sampling_config)
        
        shape = (1, 4, 8, 8, 8)
        samples1 = sampler.sample(model, shape)
        
        # Reset seed and sample again
        torch.manual_seed(42)
        samples2 = sampler.sample(model, shape)
        
        # Should be identical
        torch.testing.assert_close(samples1, samples2)


class TestPerformanceAndOptimization:
    """Test performance and optimization aspects."""
    
    def test_sampling_speed(self):
        """Test sampling speed with different configurations."""
        model = SimpleTestModel()
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        shape = (1, 4, 16, 16, 16)
        
        # Test DDPM (slower)
        ddpm_config = SamplingConfig(method=SamplingMethod.DDPM, num_inference_steps=20)
        ddpm_sampler = AdvancedSampler(scheduler, ddpm_config)
        
        start_time = time.time()
        ddpm_samples = ddpm_sampler.sample(model, shape)
        ddpm_time = time.time() - start_time
        
        # Test DDIM (faster)
        ddim_config = SamplingConfig(method=SamplingMethod.DDIM, num_inference_steps=10)
        ddim_sampler = AdvancedSampler(scheduler, ddim_config)
        
        start_time = time.time()
        ddim_samples = ddim_sampler.sample(model, shape)
        ddim_time = time.time() - start_time
        
        # DDIM should be faster (fewer steps)
        assert ddim_time < ddpm_time * 1.5  # Allow some variance
        
        # Both should produce valid samples
        assert ddpm_samples.shape == shape
        assert ddim_samples.shape == shape
    
    def test_schedule_optimization(self):
        """Test noise schedule optimization."""
        config = NoiseScheduleConfig(timesteps=100)
        scheduler = NoiseScheduler(config, device="cpu")
        
        # Generate usage statistics
        for _ in range(50):
            samples = torch.randn(2, 4, 8, 8, 8)
            scheduler.add_noise(samples)
        
        # Get initial stats
        initial_stats = scheduler.get_schedule_stats()
        
        # Optimize schedule
        scheduler.optimize_schedule()
        
        # Should not crash and should maintain valid schedule
        assert scheduler.betas.shape == (100,)
        assert torch.all(scheduler.betas > 0)
        assert torch.all(scheduler.betas < 1)


if __name__ == "__main__":
    # Run basic tests
    print("Running diffusion pipeline tests...")
    
    # Test noise scheduler
    print("Testing NoiseScheduler...")
    test_scheduler = TestNoiseScheduler()
    test_scheduler.test_linear_schedule()
    test_scheduler.test_cosine_schedule()
    test_scheduler.test_sigmoid_schedule()
    test_scheduler.test_exponential_schedule()
    test_scheduler.test_precomputed_values()
    test_scheduler.test_noise_sampling()
    test_scheduler.test_add_noise()
    test_scheduler.test_velocity_parameterization()
    test_scheduler.test_predict_start_from_noise()
    print("✓ NoiseScheduler tests passed")
    
    # Test advanced sampler
    print("Testing AdvancedSampler...")
    test_sampler = TestAdvancedSampler()
    test_sampler.setup_method()
    test_sampler.test_ddpm_sampling()
    test_sampler.test_ddim_sampling()
    test_sampler.test_conditional_sampling()
    test_sampler.test_img2img_sampling()
    print("✓ AdvancedSampler tests passed")
    
    # Test diffusion pipeline
    print("Testing Diffusion3DPipeline...")
    test_pipeline = TestDiffusion3DPipeline()
    test_pipeline.setup_method()
    test_pipeline.test_forward_process()
    test_pipeline.test_reverse_process()
    test_pipeline.test_sample_generation()
    test_pipeline.test_loss_computation()
    test_pipeline.test_pipeline_statistics()
    print("✓ Diffusion3DPipeline tests passed")
    
    # Test integration
    print("Testing integration...")
    test_integration = TestDiffusionIntegration()
    test_integration.test_different_sampling_methods()
    test_integration.test_batch_processing()
    test_integration.test_gradient_flow()
    test_integration.test_deterministic_sampling()
    print("✓ Integration tests passed")
    
    print("\n🎉 All diffusion pipeline tests passed successfully!")