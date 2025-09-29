#!/usr/bin/env python3
"""
DeepSculpt v2.0 - Complete End-to-End Pipeline

This pipeline orchestrates the complete DeepSculpt workflow:
1. 📊 Data Generation - Create synthetic 3D sculptures
2. 🔄 Data Preprocessing - Prepare data for training
3. 🧠 Model Training - Train GAN or Diffusion models
4. 🎨 Sample Generation - Generate new sculptures
5. 📈 Evaluation - Assess model performance
6. 📊 Visualization - Create beautiful 3D visualizations

Usage:
    # Run complete pipeline
    python pipeline.py --pipeline=complete --model-type=gan --epochs=50
    
    # Run specific stages
    python pipeline.py --pipeline=data-only --num-samples=1000
    python pipeline.py --pipeline=train-only --model-type=diffusion --epochs=100
    
    # Quick demo pipeline
    python pipeline.py --pipeline=demo --epochs=5
"""

import argparse
import sys
import os
import time
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available. Please install PyTorch.")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ Required dependency missing: {e}")
    sys.exit(1)

# Import DeepSculpt v2.0 modules
try:
    from core.data.generation.pytorch_collector import PyTorchCollector
    from core.data.generation.pytorch_sculptor import PyTorchSculptor
    from core.data.transforms.pytorch_curator import PyTorchCurator
    from core.models.model_factory import PyTorchModelFactory
    from core.training.pytorch_trainer import GANTrainer
    from core.training.diffusion_trainer import DiffusionTrainer
    from core.visualization.pytorch_visualization import PyTorchVisualizer
    from core.utils.pytorch_utils import PyTorchUtils
    from core.utils.logger import PyTorchLogger
    from core.utils.monitoring import DeepSculptMonitor
    from core.utils.performance_optimizer import PerformanceOptimizer
except ImportError as e:
    print(f"❌ DeepSculpt module import error: {e}")
    print("Make sure you're running from the deepsculpt_v2 directory")
    sys.exit(1)


class DeepSculptPipeline:
    """Complete end-to-end pipeline for DeepSculpt v2.0."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.logger = PyTorchLogger(
            log_level=config.get('log_level', 'INFO'),
            output_file=str(self.logs_dir / f"pipeline_{self.timestamp}.log")
        )
        
        self.monitor = DeepSculptMonitor(
            tracking_backend=config.get('tracking_backend', 'local'),
            experiment_name=f"deepsculpt_pipeline_{self.timestamp}",
            save_path=str(self.results_dir)
        ) if config.get('enable_monitoring', True) else None
        
        self.optimizer = PerformanceOptimizer() if config.get('enable_optimization', True) else None
        
        # Pipeline state
        self.pipeline_state = {
            'data_generated': False,
            'data_preprocessed': False,
            'model_trained': False,
            'samples_generated': False,
            'evaluation_completed': False
        }
        
        self.logger.info(f"🚀 DeepSculpt Pipeline initialized on {self.device}")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
    
    def setup_directories(self):
        """Setup pipeline directories."""
        base_dir = Path(self.config.get('output_dir', './pipeline_results'))
        self.base_dir = base_dir / f"pipeline_{self.timestamp}"
        
        # Create subdirectories
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.samples_dir = self.base_dir / "samples"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        self.visualizations_dir = self.base_dir / "visualizations"
        
        # Create all directories
        for directory in [self.data_dir, self.models_dir, self.samples_dir, 
                         self.results_dir, self.logs_dir, self.visualizations_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Pipeline directories created in: {self.base_dir}")
    
    def save_config(self):
        """Save pipeline configuration."""
        config_path = self.base_dir / "pipeline_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, indent=2)
        
        self.logger.info(f"💾 Configuration saved to: {config_path}")
    
    def stage_1_generate_data(self) -> bool:
        """Stage 1: Generate synthetic 3D sculpture data."""
        if self.pipeline_state['data_generated']:
            self.logger.info("📊 Data already generated, skipping...")
            return True
        
        self.logger.info("📊 Stage 1: Generating synthetic data...")
        
        try:
            # Configure sculptor
            sculptor_config = {
                "void_dim": self.config.get('void_dim', 64),
                "num_shapes": self.config.get('num_shapes', 5),
                "sparse_threshold": self.config.get('sparse_threshold', 0.1)
            }
            
            # Create collector
            collector = PyTorchCollector(
                sculptor_config=sculptor_config,
                output_format="pytorch",
                sparse_threshold=sculptor_config["sparse_threshold"],
                device=self.device
            )
            
            # Generate data
            num_samples = self.config.get('num_samples', 1000)
            self.logger.info(f"🎨 Generating {num_samples} samples...")
            
            start_time = time.time()
            dataset_paths = collector.create_collection(num_samples)
            generation_time = time.time() - start_time
            
            # Save dataset metadata
            metadata = {
                "num_samples": num_samples,
                "void_dim": sculptor_config["void_dim"],
                "num_shapes": sculptor_config["num_shapes"],
                "sparse_threshold": sculptor_config["sparse_threshold"],
                "generation_time_seconds": generation_time,
                "dataset_paths": dataset_paths,
                "timestamp": datetime.now().isoformat()
            }
            
            metadata_path = self.data_dir / "generation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.pipeline_state['data_generated'] = True
            self.logger.info(f"✅ Data generation completed in {generation_time:.2f}s")
            self.logger.info(f"📁 Generated {len(dataset_paths)} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data generation failed: {e}")
            return False
    
    def stage_2_preprocess_data(self) -> bool:
        """Stage 2: Preprocess and curate data for training."""
        if self.pipeline_state['data_preprocessed']:
            self.logger.info("🔄 Data already preprocessed, skipping...")
            return True
        
        self.logger.info("🔄 Stage 2: Preprocessing data...")
        
        try:
            # Create curator
            curator = PyTorchCurator(
                encoding_method=self.config.get('encoding_method', 'one_hot'),
                device=self.device,
                sparse_mode=self.config.get('sparse_mode', True)
            )
            
            # Load generation metadata to find data files
            metadata_path = self.data_dir / "generation_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                dataset_paths = metadata.get('dataset_paths', [])
            else:
                # Fallback: scan for data files
                dataset_paths = list(self.data_dir.glob("*.pt"))
            
            if not dataset_paths:
                self.logger.error("❌ No data files found for preprocessing")
                return False
            
            # Create training/validation split
            train_ratio = self.config.get('train_ratio', 0.8)
            split_idx = int(len(dataset_paths) * train_ratio)
            
            train_paths = dataset_paths[:split_idx]
            val_paths = dataset_paths[split_idx:]
            
            # Save split information
            split_info = {
                "train_paths": [str(p) for p in train_paths],
                "val_paths": [str(p) for p in val_paths],
                "train_ratio": train_ratio,
                "total_samples": len(dataset_paths),
                "train_samples": len(train_paths),
                "val_samples": len(val_paths)
            }
            
            split_path = self.data_dir / "data_split.json"
            with open(split_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            
            self.pipeline_state['data_preprocessed'] = True
            self.logger.info(f"✅ Data preprocessing completed")
            self.logger.info(f"📊 Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            return False
    
    def stage_3_train_model(self) -> bool:
        """Stage 3: Train GAN or Diffusion model."""
        if self.pipeline_state['model_trained']:
            self.logger.info("🧠 Model already trained, skipping...")
            return True
        
        model_type = self.config.get('model_type', 'gan')
        self.logger.info(f"🧠 Stage 3: Training {model_type.upper()} model...")
        
        try:
            # Load data split
            split_path = self.data_dir / "data_split.json"
            with open(split_path, 'r') as f:
                split_info = json.load(f)
            
            # Create data loader
            from torch.utils.data import DataLoader, Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, file_paths):
                    self.file_paths = file_paths
                
                def __len__(self):
                    return len(self.file_paths)
                
                def __getitem__(self, idx):
                    data = torch.load(self.file_paths[idx])
                    if isinstance(data, dict):
                        return data
                    else:
                        return {'structure': data, 'colors': torch.zeros_like(data)}
            
            train_dataset = SimpleDataset(split_info['train_paths'])
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=torch.cuda.is_available()
            )
            
            if model_type.lower() == 'gan':
                success = self._train_gan_model(train_loader)
            elif model_type.lower() == 'diffusion':
                success = self._train_diffusion_model(train_loader)
            else:
                self.logger.error(f"❌ Unknown model type: {model_type}")
                return False
            
            if success:
                self.pipeline_state['model_trained'] = True
                self.logger.info("✅ Model training completed")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            return False
    
    def _train_gan_model(self, train_loader) -> bool:
        """Train GAN model."""
        self.logger.info("🎭 Training GAN model...")
        
        try:
            # Create models
            generator = PyTorchModelFactory.create_gan_generator(
                model_type=self.config.get('gan_model_type', 'skip'),
                void_dim=self.config.get('void_dim', 64),
                noise_dim=self.config.get('noise_dim', 100),
                color_mode=1 if self.config.get('color_mode', True) else 0,
                sparse=self.config.get('sparse_mode', True)
            ).to(self.device)
            
            discriminator = PyTorchModelFactory.create_gan_discriminator(
                model_type=self.config.get('gan_model_type', 'skip'),
                void_dim=self.config.get('void_dim', 64),
                color_mode=1 if self.config.get('color_mode', True) else 0,
                sparse=self.config.get('sparse_mode', True)
            ).to(self.device)
            
            # Create optimizers
            gen_optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=self.config.get('learning_rate', 0.0002),
                betas=(self.config.get('beta1', 0.5), self.config.get('beta2', 0.999))
            )
            
            disc_optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=self.config.get('learning_rate', 0.0002),
                betas=(self.config.get('beta1', 0.5), self.config.get('beta2', 0.999))
            )
            
            # Create trainer
            trainer = GANTrainer(
                generator=generator,
                discriminator=discriminator,
                gen_optimizer=gen_optimizer,
                disc_optimizer=disc_optimizer,
                device=self.device,
                mixed_precision=self.config.get('mixed_precision', True)
            )
            
            # Start monitoring
            if self.monitor:
                self.monitor.start_training_monitoring(0)
            
            # Train
            epochs = self.config.get('epochs', 50)
            metrics = trainer.train(
                data_loader=train_loader,
                epochs=epochs,
                checkpoint_dir=self.models_dir / "checkpoints",
                snapshot_dir=self.visualizations_dir / "training_snapshots",
                snapshot_freq=self.config.get('snapshot_freq', 10)
            )
            
            # Save final models
            torch.save(generator.state_dict(), self.models_dir / "generator_final.pt")
            torch.save(discriminator.state_dict(), self.models_dir / "discriminator_final.pt")
            
            # Save training metrics
            with open(self.results_dir / "training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # End monitoring
            if self.monitor:
                self.monitor.end_training_monitoring()
            
            self.logger.info("✅ GAN training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GAN training failed: {e}")
            return False
    
    def _train_diffusion_model(self, train_loader) -> bool:
        """Train Diffusion model."""
        self.logger.info("🌊 Training Diffusion model...")
        
        try:
            # Create diffusion model
            model = PyTorchModelFactory.create_diffusion_model(
                model_type="unet3d",
                void_dim=self.config.get('void_dim', 64),
                timesteps=self.config.get('timesteps', 1000),
                sparse=self.config.get('sparse_mode', True)
            ).to(self.device)
            
            # Create noise scheduler
            from core.models.diffusion.noise_scheduler import NoiseScheduler
            noise_scheduler = NoiseScheduler(
                schedule_type=self.config.get('noise_schedule', 'linear'),
                timesteps=self.config.get('timesteps', 1000),
                beta_start=self.config.get('beta_start', 0.0001),
                beta_end=self.config.get('beta_end', 0.02)
            )
            
            # Create diffusion pipeline
            from core.models.diffusion.pipeline import Diffusion3DPipeline
            diffusion_pipeline = Diffusion3DPipeline(
                model=model,
                noise_scheduler=noise_scheduler,
                timesteps=self.config.get('timesteps', 1000)
            )
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
            
            # Create trainer
            trainer = DiffusionTrainer(
                model=model,
                diffusion_pipeline=diffusion_pipeline,
                optimizer=optimizer,
                device=self.device,
                mixed_precision=self.config.get('mixed_precision', True)
            )
            
            # Start monitoring
            if self.monitor:
                self.monitor.start_training_monitoring(0)
            
            # Train
            epochs = self.config.get('epochs', 50)
            metrics = trainer.train(
                data_loader=train_loader,
                epochs=epochs,
                checkpoint_dir=self.models_dir / "checkpoints"
            )
            
            # Save final model
            torch.save({
                'model_state_dict': model.state_dict(),
                'noise_scheduler': noise_scheduler,
                'config': {
                    'void_dim': self.config.get('void_dim', 64),
                    'timesteps': self.config.get('timesteps', 1000),
                    'sparse': self.config.get('sparse_mode', True)
                }
            }, self.models_dir / "diffusion_final.pt")
            
            # Save training metrics
            with open(self.results_dir / "training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # End monitoring
            if self.monitor:
                self.monitor.end_training_monitoring()
            
            self.logger.info("✅ Diffusion training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Diffusion training failed: {e}")
            return False
    
    def stage_4_generate_samples(self) -> bool:
        """Stage 4: Generate samples from trained model."""
        if self.pipeline_state['samples_generated']:
            self.logger.info("🎨 Samples already generated, skipping...")
            return True
        
        self.logger.info("🎨 Stage 4: Generating samples...")
        
        try:
            model_type = self.config.get('model_type', 'gan')
            num_samples = self.config.get('num_eval_samples', 10)
            
            if model_type.lower() == 'gan':
                success = self._generate_gan_samples(num_samples)
            elif model_type.lower() == 'diffusion':
                success = self._generate_diffusion_samples(num_samples)
            else:
                self.logger.error(f"❌ Unknown model type: {model_type}")
                return False
            
            if success:
                self.pipeline_state['samples_generated'] = True
                self.logger.info("✅ Sample generation completed")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Sample generation failed: {e}")
            return False
    
    def _generate_gan_samples(self, num_samples: int) -> bool:
        """Generate samples from GAN model."""
        try:
            # Load generator
            generator_path = self.models_dir / "generator_final.pt"
            if not generator_path.exists():
                self.logger.error("❌ Generator model not found")
                return False
            
            generator = PyTorchModelFactory.create_gan_generator(
                model_type=self.config.get('gan_model_type', 'skip'),
                void_dim=self.config.get('void_dim', 64),
                noise_dim=self.config.get('noise_dim', 100),
                color_mode=1 if self.config.get('color_mode', True) else 0,
                sparse=self.config.get('sparse_mode', True)
            ).to(self.device)
            
            generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            generator.eval()
            
            # Generate samples
            samples = []
            with torch.no_grad():
                for i in range(num_samples):
                    noise = torch.randn(1, self.config.get('noise_dim', 100), device=self.device)
                    sample = generator(noise)
                    samples.append(sample.cpu())
                    
                    # Save individual sample
                    sample_path = self.samples_dir / f"gan_sample_{i:04d}.pt"
                    torch.save(sample.cpu(), sample_path)
            
            self.logger.info(f"✅ Generated {num_samples} GAN samples")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GAN sample generation failed: {e}")
            return False
    
    def _generate_diffusion_samples(self, num_samples: int) -> bool:
        """Generate samples from Diffusion model."""
        try:
            # Load diffusion model
            model_path = self.models_dir / "diffusion_final.pt"
            if not model_path.exists():
                self.logger.error("❌ Diffusion model not found")
                return False
            
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            model = PyTorchModelFactory.create_diffusion_model(
                model_type="unet3d",
                void_dim=config['void_dim'],
                timesteps=config.get('timesteps', 1000),
                sparse=config.get('sparse', True)
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create diffusion pipeline
            from core.models.diffusion.pipeline import Diffusion3DPipeline
            noise_scheduler = checkpoint['noise_scheduler']
            diffusion_pipeline = Diffusion3DPipeline(
                model=model,
                noise_scheduler=noise_scheduler,
                timesteps=config.get('timesteps', 1000)
            )
            
            # Generate samples
            samples = []
            num_steps = self.config.get('diffusion_steps', 50)
            
            for i in range(num_samples):
                self.logger.info(f"🌊 Generating diffusion sample {i+1}/{num_samples}")
                
                shape = (1, 1 if not config.get('sparse', False) else 2,
                        config['void_dim'], config['void_dim'], config['void_dim'])
                
                sample = diffusion_pipeline.sample(
                    shape=shape,
                    num_steps=num_steps,
                    device=self.device
                )
                
                samples.append(sample.cpu())
                
                # Save individual sample
                sample_path = self.samples_dir / f"diffusion_sample_{i:04d}.pt"
                torch.save(sample.cpu(), sample_path)
            
            self.logger.info(f"✅ Generated {num_samples} diffusion samples")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Diffusion sample generation failed: {e}")
            return False
    
    def stage_5_evaluate_model(self) -> bool:
        """Stage 5: Evaluate model performance."""
        if self.pipeline_state['evaluation_completed']:
            self.logger.info("📈 Evaluation already completed, skipping...")
            return True
        
        self.logger.info("📈 Stage 5: Evaluating model performance...")
        
        try:
            # Load generated samples
            sample_files = list(self.samples_dir.glob("*.pt"))
            if not sample_files:
                self.logger.error("❌ No samples found for evaluation")
                return False
            
            # Calculate evaluation metrics
            metrics = {}
            
            # Load samples and calculate statistics
            all_samples = []
            for sample_file in sample_files:
                sample = torch.load(sample_file)
                all_samples.append(sample)
            
            if all_samples:
                # Stack all samples
                stacked_samples = torch.cat(all_samples, dim=0)
                
                # Calculate basic statistics
                metrics['num_samples'] = len(all_samples)
                metrics['mean_value'] = float(stacked_samples.mean())
                metrics['std_value'] = float(stacked_samples.std())
                metrics['min_value'] = float(stacked_samples.min())
                metrics['max_value'] = float(stacked_samples.max())
                metrics['sparsity'] = float((stacked_samples == 0).float().mean())
                
                # Calculate diversity (average pairwise distance)
                if len(all_samples) > 1:
                    distances = []
                    for i in range(len(all_samples)):
                        for j in range(i+1, len(all_samples)):
                            dist = torch.nn.functional.mse_loss(all_samples[i], all_samples[j])
                            distances.append(float(dist))
                    
                    metrics['diversity_score'] = float(np.mean(distances))
                
                # Model-specific metrics
                model_type = self.config.get('model_type', 'gan')
                metrics['model_type'] = model_type
                metrics['void_dim'] = self.config.get('void_dim', 64)
                metrics['device'] = str(self.device)
            
            # Save evaluation results
            eval_path = self.results_dir / "evaluation_metrics.json"
            with open(eval_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.pipeline_state['evaluation_completed'] = True
            self.logger.info("✅ Model evaluation completed")
            self.logger.info(f"📊 Evaluation metrics: {metrics}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return False
    
    def stage_6_create_visualizations(self) -> bool:
        """Stage 6: Create visualizations of results."""
        self.logger.info("🎨 Stage 6: Creating visualizations...")
        
        try:
            # Create visualizer
            visualizer = PyTorchVisualizer(
                backend=self.config.get('viz_backend', 'plotly'),
                device=self.device
            )
            
            # Load and visualize samples
            sample_files = list(self.samples_dir.glob("*.pt"))[:5]  # Visualize first 5 samples
            
            for i, sample_file in enumerate(sample_files):
                try:
                    sample = torch.load(sample_file)
                    
                    # Create visualization
                    viz_path = self.visualizations_dir / f"sample_{i:04d}.png"
                    visualizer.plot_sculpture(sample.squeeze(), save_path=str(viz_path))
                    
                    self.logger.info(f"🎨 Created visualization: {viz_path}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to visualize {sample_file}: {e}")
            
            # Create training metrics plot if available
            metrics_file = self.results_dir / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                self._plot_training_metrics(metrics)
            
            self.logger.info("✅ Visualizations created")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Visualization creation failed: {e}")
            return False
    
    def _plot_training_metrics(self, metrics: Dict[str, Any]):
        """Plot training metrics."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot losses
            if 'gen_loss' in metrics and 'disc_loss' in metrics:
                plt.subplot(2, 2, 1)
                plt.plot(metrics['gen_loss'], label='Generator Loss')
                plt.plot(metrics['disc_loss'], label='Discriminator Loss')
                plt.title('Training Losses')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            elif 'loss' in metrics:
                plt.subplot(2, 2, 1)
                plt.plot(metrics['loss'], label='Loss')
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            # Plot other metrics if available
            if 'epoch_times' in metrics:
                plt.subplot(2, 2, 2)
                plt.plot(metrics['epoch_times'])
                plt.title('Epoch Times')
                plt.xlabel('Epoch')
                plt.ylabel('Time (seconds)')
                plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.visualizations_dir / "training_metrics.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Training metrics plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to plot training metrics: {e}")
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete end-to-end pipeline."""
        self.logger.info("🚀 Starting complete DeepSculpt pipeline...")
        
        # Save configuration
        self.save_config()
        
        # Apply automatic optimizations
        if self.optimizer:
            optimizations = self.optimizer.apply_automatic_optimizations()
            if optimizations:
                self.logger.info(f"⚡ Applied optimizations: {optimizations}")
        
        # Run all stages
        stages = [
            ("Data Generation", self.stage_1_generate_data),
            ("Data Preprocessing", self.stage_2_preprocess_data),
            ("Model Training", self.stage_3_train_model),
            ("Sample Generation", self.stage_4_generate_samples),
            ("Model Evaluation", self.stage_5_evaluate_model),
            ("Visualization", self.stage_6_create_visualizations)
        ]
        
        start_time = time.time()
        
        for stage_name, stage_func in stages:
            self.logger.info(f"🔄 Running {stage_name}...")
            
            stage_start = time.time()
            success = stage_func()
            stage_time = time.time() - stage_start
            
            if success:
                self.logger.info(f"✅ {stage_name} completed in {stage_time:.2f}s")
            else:
                self.logger.error(f"❌ {stage_name} failed")
                return False
        
        total_time = time.time() - start_time
        
        # Create final summary
        self._create_pipeline_summary(total_time)
        
        # Shutdown monitoring
        if self.monitor:
            self.monitor.shutdown()
        
        self.logger.info(f"🎉 Complete pipeline finished in {total_time:.2f}s")
        self.logger.info(f"📁 Results available in: {self.base_dir}")
        
        return True
    
    def _create_pipeline_summary(self, total_time: float):
        """Create pipeline execution summary."""
        summary = {
            "pipeline_config": self.config,
            "execution_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "pipeline_state": self.pipeline_state,
            "output_directories": {
                "base": str(self.base_dir),
                "data": str(self.data_dir),
                "models": str(self.models_dir),
                "samples": str(self.samples_dir),
                "results": str(self.results_dir),
                "visualizations": str(self.visualizations_dir)
            }
        }
        
        # Add file counts
        summary["output_files"] = {
            "data_files": len(list(self.data_dir.glob("*.pt"))),
            "model_files": len(list(self.models_dir.glob("*.pt"))),
            "sample_files": len(list(self.samples_dir.glob("*.pt"))),
            "visualization_files": len(list(self.visualizations_dir.glob("*.png")))
        }
        
        # Save summary
        summary_path = self.base_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📋 Pipeline summary saved: {summary_path}")


def create_config_from_args(args) -> Dict[str, Any]:
    """Create configuration dictionary from command line arguments."""
    config = {
        # Model configuration
        'model_type': args.model_type,
        'gan_model_type': args.gan_model_type,
        'void_dim': args.void_dim,
        'noise_dim': args.noise_dim,
        
        # Training configuration
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'mixed_precision': args.mixed_precision,
        
        # Data configuration
        'num_samples': args.num_samples,
        'num_shapes': args.num_shapes,
        'sparse_mode': args.sparse,
        'sparse_threshold': args.sparse_threshold,
        
        # Pipeline configuration
        'output_dir': args.output_dir,
        'log_level': args.log_level,
        'enable_monitoring': args.enable_monitoring,
        'enable_optimization': args.enable_optimization,
        
        # Evaluation configuration
        'num_eval_samples': args.num_eval_samples,
        'viz_backend': args.viz_backend
    }
    
    return config


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="DeepSculpt v2.0 - Complete End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Pipeline type
    parser.add_argument('--pipeline', default='complete',
                       choices=['complete', 'data-only', 'train-only', 'demo'],
                       help='Pipeline type to run')
    
    # Model configuration
    parser.add_argument('--model-type', default='gan',
                       choices=['gan', 'diffusion'],
                       help='Type of model to train')
    parser.add_argument('--gan-model-type', default='skip',
                       choices=['simple', 'complex', 'skip', 'monochrome'],
                       help='GAN model architecture')
    parser.add_argument('--void-dim', type=int, default=64,
                       help='3D voxel space dimension')
    parser.add_argument('--noise-dim', type=int, default=100,
                       help='Noise vector dimension')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    
    # Data configuration
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of data samples to generate')
    parser.add_argument('--num-shapes', type=int, default=5,
                       help='Number of shapes per sculpture')
    parser.add_argument('--sparse', action='store_true',
                       help='Enable sparse tensor mode')
    parser.add_argument('--sparse-threshold', type=float, default=0.1,
                       help='Sparse tensor threshold')
    
    # Pipeline configuration
    parser.add_argument('--output-dir', default='./pipeline_results',
                       help='Output directory for results')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--enable-monitoring', action='store_true', default=True,
                       help='Enable performance monitoring')
    parser.add_argument('--enable-optimization', action='store_true', default=True,
                       help='Enable automatic optimizations')
    
    # Evaluation configuration
    parser.add_argument('--num-eval-samples', type=int, default=10,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--viz-backend', default='plotly',
                       choices=['matplotlib', 'plotly'],
                       help='Visualization backend')
    
    args = parser.parse_args()
    
    # Adjust configuration for pipeline type
    if args.pipeline == 'demo':
        args.num_samples = 50
        args.epochs = 5
        args.batch_size = 8
        args.void_dim = 32
        args.num_eval_samples = 3
    elif args.pipeline == 'data-only':
        args.epochs = 0  # Skip training
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create and run pipeline
    pipeline = DeepSculptPipeline(config)
    
    try:
        if args.pipeline == 'complete' or args.pipeline == 'demo':
            success = pipeline.run_complete_pipeline()
        elif args.pipeline == 'data-only':
            success = (pipeline.stage_1_generate_data() and 
                      pipeline.stage_2_preprocess_data() and
                      pipeline.stage_6_create_visualizations())
        elif args.pipeline == 'train-only':
            success = (pipeline.stage_3_train_model() and
                      pipeline.stage_4_generate_samples() and
                      pipeline.stage_5_evaluate_model() and
                      pipeline.stage_6_create_visualizations())
        else:
            print(f"❌ Unknown pipeline type: {args.pipeline}")
            return 1
        
        if success:
            print(f"\n🎉 Pipeline '{args.pipeline}' completed successfully!")
            print(f"📁 Results available in: {pipeline.base_dir}")
            return 0
        else:
            print(f"\n❌ Pipeline '{args.pipeline}' failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())