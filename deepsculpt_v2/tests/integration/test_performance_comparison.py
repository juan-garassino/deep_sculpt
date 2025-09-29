#!/usr/bin/env python3
"""
Performance Comparison Tests for DeepSculpt v2.0

Tests comparing PyTorch implementation performance against baselines
and validating memory usage and GPU utilization.
"""

import pytest
import torch
import time
import psutil
import gc
from typing import Dict, List, Tuple
import numpy as np

# Import DeepSculpt v2.0 modules
from core.models.model_factory import PyTorchModelFactory
from core.data.generation.pytorch_collector import PyTorchCollector
from core.data.generation.pytorch_sculptor import PyTorchSculptor
from core.utils.pytorch_utils import PyTorchUtils


class TestPerformanceComparison:
    """Performance comparison and validation tests."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_configs = [
            {"void_dim": 32, "batch_size": 4, "name": "small"},
            {"void_dim": 64, "batch_size": 2, "name": "medium"},
        ]
        
        # Add large config only if we have enough memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 8e9:  # 8GB+
                self.test_configs.append({"void_dim": 128, "batch_size": 1, "name": "large"})
        
        yield
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @pytest.mark.benchmark
    def test_model_inference_performance(self):
        """Benchmark model inference performance across different configurations."""
        print("Benchmarking model inference performance...")
        
        results = {}
        
        for config in self.test_configs:
            print(f"\nTesting {config['name']} configuration...")
            
            # Create model
            model = PyTorchModelFactory.create_gan_generator(
                model_type="skip",
                void_dim=config["void_dim"],
                noise_dim=100,
                sparse=False
            ).to(self.device)
            
            model.eval()
            
            # Warm up
            with torch.no_grad():
                warmup_noise = torch.randn(1, 100, device=self.device)
                _ = model(warmup_noise)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark inference
            batch_size = config["batch_size"]
            num_iterations = 10
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    noise = torch.randn(batch_size, 100, device=self.device)
                    output = model(noise)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            samples_per_second = (batch_size * num_iterations) / total_time
            
            results[config['name']] = {
                'avg_time_per_batch': avg_time_per_batch,
                'samples_per_second': samples_per_second,
                'total_time': total_time,
                'void_dim': config['void_dim'],
                'batch_size': batch_size
            }
            
            print(f"  Average time per batch: {avg_time_per_batch:.4f}s")
            print(f"  Samples per second: {samples_per_second:.2f}")
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Validate performance scaling
        if len(results) >= 2:
            small_sps = results['small']['samples_per_second']
            medium_sps = results['medium']['samples_per_second']
            
            # Medium should be slower than small (due to larger tensors)
            assert medium_sps < small_sps * 2, "Performance scaling seems incorrect"
            print("✓ Performance scaling validated")
        
        print("✅ Model inference performance benchmark completed")
        return results
    
    @pytest.mark.benchmark
    def test_memory_usage_validation(self):
        """Validate memory usage patterns and optimization."""
        print("Testing memory usage validation...")
        
        memory_results = {}
        
        for config in self.test_configs:
            print(f"\nTesting memory usage for {config['name']} configuration...")
            
            # Measure initial memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_gpu_memory = torch.cuda.memory_allocated()
            
            initial_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Create model and data
            model = PyTorchModelFactory.create_gan_generator(
                model_type="skip",
                void_dim=config["void_dim"],
                noise_dim=100,
                sparse=False
            ).to(self.device)
            
            # Create test data
            batch_size = config["batch_size"]
            noise = torch.randn(batch_size, 100, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                output = model(noise)
            
            # Measure peak memory
            if torch.cuda.is_available():
                peak_gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_used = (peak_gpu_memory - initial_gpu_memory) / 1024 / 1024  # MB
            else:
                gpu_memory_used = 0
            
            peak_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            cpu_memory_used = peak_cpu_memory - initial_cpu_memory
            
            memory_results[config['name']] = {
                'gpu_memory_mb': gpu_memory_used,
                'cpu_memory_mb': cpu_memory_used,
                'void_dim': config['void_dim'],
                'batch_size': batch_size,
                'output_size_mb': output.numel() * output.element_size() / 1024 / 1024
            }
            
            print(f"  GPU memory used: {gpu_memory_used:.1f} MB")
            print(f"  CPU memory used: {cpu_memory_used:.1f} MB")
            print(f"  Output tensor size: {memory_results[config['name']]['output_size_mb']:.1f} MB")
            
            # Validate memory usage is reasonable
            expected_output_size = batch_size * config['void_dim']**3 * 4 / 1024 / 1024  # 4 bytes per float32
            assert memory_results[config['name']]['output_size_mb'] >= expected_output_size * 0.8
            assert memory_results[config['name']]['output_size_mb'] <= expected_output_size * 1.2
            
            # Cleanup
            del model, noise, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        print("✓ Memory usage patterns validated")
        print("✅ Memory usage validation completed")
        return memory_results
    
    @pytest.mark.benchmark
    def test_sparse_vs_dense_performance(self):
        """Compare sparse vs dense tensor performance."""
        print("Testing sparse vs dense tensor performance...")
        
        void_dim = 64
        batch_size = 2
        
        # Test data generation performance
        print("\nTesting data generation performance...")
        
        # Dense sculptor
        dense_sculptor = PyTorchSculptor(
            void_dim=void_dim,
            device=self.device,
            sparse_mode=False
        )
        
        # Sparse sculptor
        sparse_sculptor = PyTorchSculptor(
            void_dim=void_dim,
            device=self.device,
            sparse_mode=True
        )
        
        # Benchmark dense generation
        start_time = time.time()
        dense_samples = []
        for _ in range(5):
            structure, colors = dense_sculptor.generate_sculpture()
            dense_samples.append((structure, colors))
        dense_time = time.time() - start_time
        
        # Benchmark sparse generation
        start_time = time.time()
        sparse_samples = []
        for _ in range(5):
            structure, colors = sparse_sculptor.generate_sculpture()
            sparse_samples.append((structure, colors))
        sparse_time = time.time() - start_time
        
        print(f"Dense generation time: {dense_time:.4f}s")
        print(f"Sparse generation time: {sparse_time:.4f}s")
        
        # Test model performance with sparse vs dense
        print("\nTesting model performance...")
        
        # Create models
        dense_model = PyTorchModelFactory.create_gan_generator(
            model_type="simple",
            void_dim=void_dim,
            noise_dim=100,
            sparse=False
        ).to(self.device)
        
        sparse_model = PyTorchModelFactory.create_gan_generator(
            model_type="simple",
            void_dim=void_dim,
            noise_dim=100,
            sparse=True
        ).to(self.device)
        
        # Benchmark inference
        noise = torch.randn(batch_size, 100, device=self.device)
        
        # Dense model
        dense_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = dense_model(noise)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        dense_inference_time = time.time() - start_time
        
        # Sparse model
        sparse_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = sparse_model(noise)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        sparse_inference_time = time.time() - start_time
        
        print(f"Dense model inference time: {dense_inference_time:.4f}s")
        print(f"Sparse model inference time: {sparse_inference_time:.4f}s")
        
        # Memory comparison
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Dense memory
            with torch.no_grad():
                dense_output = dense_model(noise)
            dense_memory = torch.cuda.memory_allocated()
            
            torch.cuda.empty_cache()
            
            # Sparse memory
            with torch.no_grad():
                sparse_output = sparse_model(noise)
            sparse_memory = torch.cuda.memory_allocated()
            
            print(f"Dense model memory: {dense_memory / 1024 / 1024:.1f} MB")
            print(f"Sparse model memory: {sparse_memory / 1024 / 1024:.1f} MB")
        
        results = {
            'dense_generation_time': dense_time,
            'sparse_generation_time': sparse_time,
            'dense_inference_time': dense_inference_time,
            'sparse_inference_time': sparse_inference_time
        }
        
        print("✓ Sparse vs dense performance comparison completed")
        return results
    
    @pytest.mark.benchmark
    @pytest.mark.gpu
    def test_gpu_utilization_monitoring(self):
        """Monitor and validate GPU utilization during operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        print("Testing GPU utilization monitoring...")
        
        # Create a computationally intensive model
        model = PyTorchModelFactory.create_gan_generator(
            model_type="complex",
            void_dim=64,
            noise_dim=100,
            sparse=False
        ).to("cuda")
        
        discriminator = PyTorchModelFactory.create_gan_discriminator(
            model_type="complex",
            void_dim=64,
            sparse=False
        ).to("cuda")
        
        # Setup optimizers
        gen_optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        # Monitor GPU utilization during training simulation
        print("Simulating training to monitor GPU utilization...")
        
        batch_size = 4
        num_iterations = 20
        
        gpu_memory_usage = []
        
        for i in range(num_iterations):
            # Generator training step
            gen_optimizer.zero_grad()
            
            noise = torch.randn(batch_size, 100, device="cuda")
            fake_data = model(noise)
            
            # Simulate discriminator feedback
            disc_fake = discriminator(fake_data)
            gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_fake, torch.ones_like(disc_fake)
            )
            
            gen_loss.backward()
            gen_optimizer.step()
            
            # Discriminator training step
            disc_optimizer.zero_grad()
            
            # Real data (simulated)
            real_data = torch.randn_like(fake_data)
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data.detach())
            
            disc_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_real, torch.ones_like(disc_real)
                ) +
                torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_fake, torch.zeros_like(disc_fake)
                )
            ) / 2
            
            disc_loss.backward()
            disc_optimizer.step()
            
            # Record GPU memory usage
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_usage.append(current_memory)
            
            if i % 5 == 0:
                print(f"  Iteration {i}: GPU memory {current_memory:.1f} MB")
        
        # Analyze GPU utilization
        avg_memory = np.mean(gpu_memory_usage)
        max_memory = np.max(gpu_memory_usage)
        memory_std = np.std(gpu_memory_usage)
        
        print(f"\nGPU Memory Statistics:")
        print(f"  Average: {avg_memory:.1f} MB")
        print(f"  Maximum: {max_memory:.1f} MB")
        print(f"  Std Dev: {memory_std:.1f} MB")
        
        # Validate GPU utilization
        assert max_memory > 0, "No GPU memory usage detected"
        assert memory_std < max_memory * 0.5, "GPU memory usage too unstable"
        
        # Test memory cleanup
        del model, discriminator, gen_optimizer, disc_optimizer
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  Final memory after cleanup: {final_memory:.1f} MB")
        
        print("✓ GPU utilization monitoring completed")
        
        return {
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'memory_std_mb': memory_std,
            'final_memory_mb': final_memory
        }
    
    @pytest.mark.benchmark
    def test_data_loading_performance(self):
        """Benchmark data loading and preprocessing performance."""
        print("Testing data loading performance...")
        
        # Test different data loading configurations
        configs = [
            {"num_workers": 0, "batch_size": 4, "name": "single_thread"},
            {"num_workers": 2, "batch_size": 4, "name": "multi_thread"},
            {"num_workers": 0, "batch_size": 8, "name": "large_batch"},
        ]
        
        results = {}
        
        for config in configs:
            print(f"\nTesting {config['name']} configuration...")
            
            # Create dataset
            sculptor_config = {
                "void_dim": 32,
                "num_shapes": 3,
                "sparse_threshold": 0.1
            }
            
            collector = PyTorchCollector(
                sculptor_config=sculptor_config,
                device=self.device
            )
            
            dataset = collector.create_streaming_dataset(50)
            
            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                pin_memory=torch.cuda.is_available()
            )
            
            # Benchmark data loading
            start_time = time.time()
            
            batch_count = 0
            for batch in data_loader:
                # Simulate some processing
                _ = batch['structure'].mean()
                _ = batch['colors'].mean()
                batch_count += 1
                
                if batch_count >= 10:  # Limit for testing
                    break
            
            end_time = time.time()
            
            total_time = end_time - start_time
            batches_per_second = batch_count / total_time
            samples_per_second = batches_per_second * config["batch_size"]
            
            results[config['name']] = {
                'total_time': total_time,
                'batches_per_second': batches_per_second,
                'samples_per_second': samples_per_second,
                'num_workers': config["num_workers"],
                'batch_size': config["batch_size"]
            }
            
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Batches per second: {batches_per_second:.2f}")
            print(f"  Samples per second: {samples_per_second:.2f}")
        
        # Validate that multi-threading helps (if available)
        if "single_thread" in results and "multi_thread" in results:
            single_sps = results["single_thread"]["samples_per_second"]
            multi_sps = results["multi_thread"]["samples_per_second"]
            
            print(f"\nMulti-threading speedup: {multi_sps / single_sps:.2f}x")
            
            # Multi-threading should provide some benefit (at least not hurt)
            assert multi_sps >= single_sps * 0.8, "Multi-threading significantly hurts performance"
        
        print("✓ Data loading performance benchmark completed")
        return results
    
    @pytest.mark.benchmark
    def test_training_step_performance(self):
        """Benchmark individual training step performance."""
        print("Testing training step performance...")
        
        # Create models
        generator = PyTorchModelFactory.create_gan_generator(
            model_type="skip",
            void_dim=32,
            noise_dim=100,
            sparse=False
        ).to(self.device)
        
        discriminator = PyTorchModelFactory.create_gan_discriminator(
            model_type="skip",
            void_dim=32,
            sparse=False
        ).to(self.device)
        
        # Create optimizers
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        # Create sample data
        batch_size = 4
        real_data = torch.randn(batch_size, 1, 32, 32, 32, device=self.device)
        
        # Benchmark training steps
        num_steps = 20
        step_times = []
        
        for i in range(num_steps):
            step_start = time.time()
            
            # Generator step
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size, 100, device=self.device)
            fake_data = generator(noise)
            disc_fake = discriminator(fake_data)
            gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_fake, torch.ones_like(disc_fake)
            )
            gen_loss.backward()
            gen_optimizer.step()
            
            # Discriminator step
            disc_optimizer.zero_grad()
            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data.detach())
            
            disc_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_real, torch.ones_like(disc_real)
                ) +
                torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_fake, torch.zeros_like(disc_fake)
                )
            ) / 2
            
            disc_loss.backward()
            disc_optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            step_end = time.time()
            step_times.append(step_end - step_start)
            
            if i % 5 == 0:
                print(f"  Step {i}: {step_times[-1]:.4f}s")
        
        # Analyze performance
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)
        min_step_time = np.min(step_times)
        max_step_time = np.max(step_times)
        
        print(f"\nTraining Step Performance:")
        print(f"  Average: {avg_step_time:.4f}s")
        print(f"  Std Dev: {std_step_time:.4f}s")
        print(f"  Min: {min_step_time:.4f}s")
        print(f"  Max: {max_step_time:.4f}s")
        
        # Validate performance consistency
        assert std_step_time < avg_step_time * 0.5, "Training step times too inconsistent"
        assert max_step_time < avg_step_time * 3, "Some training steps took too long"
        
        print("✓ Training step performance benchmark completed")
        
        return {
            'avg_step_time': avg_step_time,
            'std_step_time': std_step_time,
            'min_step_time': min_step_time,
            'max_step_time': max_step_time,
            'steps_per_second': 1.0 / avg_step_time
        }


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])