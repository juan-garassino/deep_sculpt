#!/usr/bin/env python3
"""
DeepSculpt v2.0 - Local Training Script

Complete workflow: Dataset Creation → Training → Generation

This script demonstrates the complete DeepSculpt pipeline with:
- Dataset generation (small dataset for local testing)
- Model training (minimal epochs)
- Sample generation from trained model
- Visualization of results

For production training: Migrate to Colab and increase parameters
"""

import sys
import os
from pathlib import Path
import time
import json
import argparse

# Core imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DeepSculpt modules
from core.data.generation.pytorch_sculptor import PyTorchSculptor
from core.data.generation.pytorch_collector import PyTorchCollector
from core.models.model_factory import PyTorchModelFactory
from core.training.pytorch_trainer import GANTrainer
from core.visualization.pytorch_visualization import PyTorchVisualizer
from core.utils.pytorch_utils import PyTorchUtils


class SimpleDataset(Dataset):
    """Simple dataset for loading generated samples."""
    
    def __init__(self, file_paths, device='cpu'):
        self.file_paths = file_paths
        self.device = device
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load structure
        structure = torch.load(self.file_paths[idx])
        
        # Ensure correct shape (add batch and channel dims if needed)
        if structure.dim() == 3:
            structure = structure.unsqueeze(0)  # Add channel dimension
        
        # Convert to float and normalize to [0, 1] or [-1, 1]
        structure = structure.float()
        if structure.max() > 1.0:
            structure = structure / 255.0  # Normalize if needed
        
        return structure.to(self.device)


def print_system_info():
    """Print system and PyTorch information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU")
    print()


def get_config(args):
    """Get configuration for training."""
    config = {
        # Data generation
        'void_dim': args.void_dim,
        'num_samples': args.num_samples,
        'num_shapes': 3,
        
        # Model architecture
        'model_type': args.model_type,
        'noise_dim': args.noise_dim,
        'color_mode': args.color_mode,
        
        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        
        # Generation
        'num_eval_samples': args.num_eval_samples,
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sparse_mode': args.sparse,
        'mixed_precision': args.mixed_precision,
        
        # Paths
        'output_dir': args.output_dir,
    }
    
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    return config


def generate_dataset(config):
    """Generate synthetic 3D sculpture dataset."""
    print("=" * 60)
    print("DATASET GENERATION")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Data directory: {data_dir}")
    print()
    
    # Configure sculptor
    sculptor_config = {
        'void_dim': config['void_dim'],
        'edges': (1, 0.3, 0.5),
        'planes': (1, 0.3, 0.5),
        'pipes': (1, 0.3, 0.5),
        'grid': (1, 4),
        'step': 1,
    }
    
    print("Sculptor configuration:")
    for key, value in sculptor_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create collector
    collector = PyTorchCollector(
        sculptor_config=sculptor_config,
        output_format='pytorch',
        base_dir=str(data_dir),
        device=config['device'],
        sparse_mode=config['sparse_mode'],
        verbose=True,
    )
    
    print(f"✅ Collector created on device: {config['device']}")
    print()
    
    # Generate dataset
    print(f"🎨 Generating {config['num_samples']} samples...")
    print()
    start_time = time.time()
    
    sample_paths = collector.create_collection(
        num_samples=config['num_samples'],
        batch_size=config['batch_size'],
        dynamic_batching=False,
    )
    
    generation_time = time.time() - start_time
    
    print()
    print(f"✅ Generated {len(sample_paths)} samples in {generation_time:.2f}s")
    print(f"⏱️  Average: {generation_time/len(sample_paths):.3f}s per sample")
    
    # Show generation stats
    stats = collector.get_generation_stats()
    print()
    print("📊 Generation Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Errors: {stats['errors']}")
    print()
    
    return sample_paths


def create_dataloader(sample_paths, config):
    """Create data loader for training."""
    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    
    # Create dataset and dataloader
    dataset = SimpleDataset(sample_paths, device=config['device'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    
    print(f"✅ Created dataset with {len(dataset)} samples")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🔄 Number of batches: {len(dataloader)}")
    
    # Test loading a batch
    test_batch = next(iter(dataloader))
    print()
    print(f"📊 Test batch shape: {test_batch.shape}")
    print(f"📊 Test batch dtype: {test_batch.dtype}")
    print(f"📊 Test batch device: {test_batch.device}")
    print()
    
    return dataloader


def create_models(config):
    """Create GAN generator and discriminator models."""
    print("=" * 60)
    print("MODEL CREATION")
    print("=" * 60)
    
    # Create model factory
    factory = PyTorchModelFactory(device=config['device'])
    print(f"🏭 Model factory created on device: {config['device']}")
    print()
    
    # Create generator
    generator = factory.create_gan_generator(
        model_type=config['model_type'],
        void_dim=config['void_dim'],
        noise_dim=config['noise_dim'],
        color_mode=config['color_mode'],
        sparse=config['sparse_mode'],
    )
    
    print("✅ Generator created")
    gen_info = factory.get_model_info(generator)
    print(f"  Parameters: {gen_info['total_parameters']:,}")
    print(f"  Memory: {gen_info['memory_usage_mb']:.2f} MB")
    print()
    
    # Create discriminator
    discriminator = factory.create_gan_discriminator(
        model_type=config['model_type'],
        void_dim=config['void_dim'],
        color_mode=config['color_mode'],
        sparse=config['sparse_mode'],
    )
    
    print("✅ Discriminator created")
    disc_info = factory.get_model_info(discriminator)
    print(f"  Parameters: {disc_info['total_parameters']:,}")
    print(f"  Memory: {disc_info['memory_usage_mb']:.2f} MB")
    print()
    print(f"📊 Total model memory: {gen_info['memory_usage_mb'] + disc_info['memory_usage_mb']:.2f} MB")
    print()
    
    # Test generator forward pass
    print("🧪 Testing generator...")
    test_noise = torch.randn(1, config['noise_dim'], device=config['device'])
    with torch.no_grad():
        test_output = generator(test_noise)
    print(f"  Input shape: {test_noise.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✅ Generator test passed")
    print()
    
    return generator, discriminator


def train_model(generator, discriminator, dataloader, config):
    """Train the GAN model."""
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Import TrainingConfig
    from core.training.pytorch_trainer import TrainingConfig
    
    # Create log directory
    output_dir = Path(config['output_dir'])
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        mixed_precision=config['mixed_precision'],
        gradient_clip=1.0,
        checkpoint_freq=1,
        log_dir=str(log_dir),
        use_tensorboard=False,  # Disable TensorBoard
        use_wandb=False,  # Disable Wandb
        use_mlflow=False,  # Disable MLflow
    )
    
    # Create optimizers
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
    )
    
    print("✅ Optimizers created")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Beta1: {config['beta1']}, Beta2: {config['beta2']}")
    print()
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        config=training_config,
        device=config['device'],
        noise_dim=config['noise_dim'],
    )
    
    print("✅ Trainer created")
    print(f"  Device: {config['device']}")
    print(f"  Mixed precision: {config['mixed_precision']}")
    print()
    
    # Create checkpoint directory
    output_dir = Path(config['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"🚀 Starting training for {config['epochs']} epochs...")
    print()
    
    # Train the model
    start_time = time.time()
    
    try:
        # Initialize metrics storage
        all_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
        }
        
        # Training loop
        for epoch in range(config['epochs']):
            print(f"Epoch {epoch+1}/{config['epochs']}")
            
            # Train for one epoch
            epoch_metrics = trainer.train_epoch(dataloader)
            
            # Store metrics
            for key in all_metrics.keys():
                if key in epoch_metrics:
                    all_metrics[key].append(epoch_metrics[key])
            
            # Print epoch summary
            print(f"  Gen Loss: {epoch_metrics.get('gen_loss', 0):.4f}")
            print(f"  Disc Loss: {epoch_metrics.get('disc_loss', 0):.4f}")
            print(f"  Disc Real Acc: {epoch_metrics.get('disc_real_acc', 0):.4f}")
            print(f"  Disc Fake Acc: {epoch_metrics.get('disc_fake_acc', 0):.4f}")
            print()
            
            # Save checkpoint
            if (epoch + 1) % training_config.checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                trainer.save_checkpoint(
                    str(checkpoint_path),
                    epoch=epoch+1,
                    metrics=epoch_metrics,
                    is_best=False
                )
                print(f"  💾 Checkpoint saved: {checkpoint_path}")
                print()
        
        training_time = time.time() - start_time
        
        print()
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"⏱️  Average: {training_time/config['epochs']:.2f}s per epoch")
        print()
        
        return all_metrics, training_time
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def plot_metrics(metrics, config):
    """Plot training metrics."""
    if not metrics:
        return
    
    output_dir = Path(config['output_dir'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    if 'gen_loss' in metrics and 'disc_loss' in metrics:
        axes[0].plot(metrics['gen_loss'], label='Generator Loss', marker='o')
        axes[0].plot(metrics['disc_loss'], label='Discriminator Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot discriminator accuracy if available
    if 'disc_real_acc' in metrics and 'disc_fake_acc' in metrics:
        axes[1].plot(metrics['disc_real_acc'], label='Real Accuracy', marker='o')
        axes[1].plot(metrics['disc_fake_acc'], label='Fake Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training metrics saved to {output_dir / 'training_metrics.png'}")


def save_models(generator, discriminator, config, training_time):
    """Save trained models and configuration."""
    print("=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    output_dir = Path(config['output_dir'])
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    generator_path = models_dir / 'generator_final.pt'
    discriminator_path = models_dir / 'discriminator_final.pt'
    
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    
    print(f"✅ Models saved:")
    print(f"  Generator: {generator_path}")
    print(f"  Discriminator: {discriminator_path}")
    print()
    
    # Save configuration
    config_path = models_dir / 'config.json'
    
    config_to_save = {
        'model_type': config['model_type'],
        'void_dim': config['void_dim'],
        'noise_dim': config['noise_dim'],
        'color_mode': config['color_mode'],
        'sparse_mode': config['sparse_mode'],
        'training': {
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'beta1': config['beta1'],
            'beta2': config['beta2'],
        },
        'training_time': training_time,
        'num_samples': config['num_samples'],
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"✅ Configuration saved: {config_path}")
    print()


def generate_samples(generator, config):
    """Generate new sculptures using the trained generator."""
    print("=" * 60)
    print("SAMPLE GENERATION")
    print("=" * 60)
    
    output_dir = Path(config['output_dir'])
    samples_dir = output_dir / 'generated_samples'
    samples_dir.mkdir(exist_ok=True)
    
    print(f"📁 Samples directory: {samples_dir}")
    print(f"🎨 Generating {config['num_eval_samples']} samples...")
    print()
    
    # Generate samples
    generator.eval()
    generated_samples = []
    
    with torch.no_grad():
        for i in range(config['num_eval_samples']):
            # Generate random noise
            noise = torch.randn(1, config['noise_dim'], device=config['device'])
            
            # Generate sample
            sample = generator(noise)
            generated_samples.append(sample.cpu())
            
            # Save sample
            sample_path = samples_dir / f'sample_{i:03d}.pt'
            torch.save(sample.cpu(), sample_path)
            
            print(f"  Sample {i+1}/{config['num_eval_samples']}: {sample.shape}")
    
    print()
    print(f"✅ Generated {len(generated_samples)} samples")
    print()
    
    return generated_samples


def visualize_samples(generated_samples, config):
    """Visualize the generated sculptures."""
    output_dir = Path(config['output_dir'])
    
    # Visualize generated samples
    fig, axes = plt.subplots(1, config['num_eval_samples'], 
                            figsize=(5*config['num_eval_samples'], 5))
    
    if config['num_eval_samples'] == 1:
        axes = [axes]
    
    for i, sample in enumerate(generated_samples):
        # Get middle slice
        sample_np = sample.squeeze().numpy()
        
        if sample_np.ndim == 4:  # (C, D, H, W)
            sample_np = sample_np[0]  # Take first channel
        
        mid_slice = sample_np.shape[0] // 2
        
        axes[i].imshow(sample_np[mid_slice], cmap='viridis')
        axes[i].set_title(f'Sample {i+1} (z={mid_slice})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generated_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to {output_dir / 'generated_samples.png'}")
    print()
    
    # Show sample statistics
    print("📊 Sample Statistics:")
    for i, sample in enumerate(generated_samples):
        sample_np = sample.squeeze().numpy()
        if sample_np.ndim == 4:
            sample_np = sample_np[0]
        
        non_zero = (sample_np != 0).sum()
        total = sample_np.size
        sparsity = 1.0 - (non_zero / total)
        
        print(f"  Sample {i+1}:")
        print(f"    Shape: {sample_np.shape}")
        print(f"    Range: [{sample_np.min():.3f}, {sample_np.max():.3f}]")
        print(f"    Non-zero voxels: {non_zero}/{total}")
        print(f"    Sparsity: {sparsity:.3f}")
    print()


def print_summary(config, training_time):
    """Print final summary."""
    print("=" * 60)
    print("🎉 COMPLETE")
    print("=" * 60)
    print(f"Output: {config['output_dir']}")
    print()


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='DeepSculpt v2.0 Local Training')
    
    # Data generation
    parser.add_argument('--void-dim', type=int, default=16,
                       help='3D grid dimension (default: 16)')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of training samples (default: 20)')
    
    # Model architecture
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'skip', 'residual'],
                       help='Model architecture type (default: simple)')
    parser.add_argument('--noise-dim', type=int, default=50,
                       help='Noise dimension (default: 50)')
    parser.add_argument('--color-mode', type=int, default=1,
                       help='Color mode: 0=monochrome, 1=color (default: 1)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    
    # Generation
    parser.add_argument('--num-eval-samples', type=int, default=3,
                       help='Number of samples to generate (default: 3)')
    
    # System
    parser.add_argument('--sparse', action='store_true',
                       help='Enable sparse mode')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default='./local_example_output',
                       help='Output directory (default: ./local_example_output)')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Get configuration
    config = get_config(args)
    
    # Generate dataset
    sample_paths = generate_dataset(config)
    
    # Create dataloader
    dataloader = create_dataloader(sample_paths, config)
    
    # Create models
    generator, discriminator = create_models(config)
    
    # Train model
    metrics, training_time = train_model(generator, discriminator, dataloader, config)
    
    # Plot metrics
    plot_metrics(metrics, config)
    
    # Save models
    save_models(generator, discriminator, config, training_time)
    
    # Generate samples
    generated_samples = generate_samples(generator, config)
    
    # Visualize samples
    visualize_samples(generated_samples, config)
    
    # Print summary
    print_summary(config, training_time)


if __name__ == '__main__':
    main()
