#!/usr/bin/env python3
"""
DeepSculpt - Unified Main Orchestrator

This is the main entry point for the DeepSculpt project, supporting both TensorFlow and PyTorch:
1. Model creation and management (TensorFlow/PyTorch)
2. Training pipeline (GAN/Diffusion)
3. Workflow automation
4. API server
5. Telegram bot interface
6. Model migration utilities
7. Diffusion model training and inference

Usage:
    # TensorFlow training (legacy)
    python main.py train --framework=tensorflow --model-type=skip --epochs=100 --data-folder=./data
    
    # PyTorch GAN training
    python main.py train --framework=pytorch --model-type=skip --epochs=100 --data-folder=./data
    
    # PyTorch diffusion training
    python main.py train-diffusion --epochs=100 --data-folder=./data --sparse
    
    # Diffusion inference
    python main.py sample-diffusion --checkpoint=./checkpoints/diffusion.pt --num-samples=10
    
    # Model migration
    python main.py migrate-model --tf-checkpoint=./tf_model --pytorch-output=./pytorch_model
    
    # API and bot services
    python main.py serve-api --port=8000
    python main.py run-bot --token=YOUR_TELEGRAM_TOKEN
    python main.py workflow --framework=pytorch --mode=development
    python main.py all --framework=pytorch --mode=production
"""

import os
import sys
import argparse
import subprocess
import threading
import time
import pandas as pd
from datetime import datetime
import signal
import uvicorn
import multiprocessing
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Framework imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

# Import DeepSculpt modules
# Use try/except to handle potential import errors gracefully
try:
    # TensorFlow modules (legacy)
    if TF_AVAILABLE:
        from models import ModelFactory as TFModelFactory
        from trainer import DeepSculptTrainer, DataFrameDataLoader, create_data_dataframe
    
    # PyTorch modules
    if PYTORCH_AVAILABLE:
        from pytorch_models import PyTorchModelFactory
        from pytorch_trainer import GANTrainer, DiffusionTrainer, BaseTrainer
        from pytorch_collector import PyTorchCollector
        from pytorch_curator import PyTorchCurator
        from pytorch_sculptor import PyTorchSculptor
        from pytorch_diffusion import Diffusion3DPipeline, NoiseScheduler
        from pytorch_workflow import PyTorchManager
        from pytorch_mlflow_tracking import PyTorchMLflowTracker
        from pytorch_utils import PyTorchUtils
        from pytorch_visualization import PyTorchVisualizer
    
    # Common modules
    from workflow import Manager, build_flow
    import api
    import bot
    
except ImportError as e:
    print(f"Error importing DeepSculpt modules: {e}")
    print("Make sure all required modules are in the same directory or in PYTHONPATH")
    sys.exit(1)


# Configure environment variables if not already set
def setup_environment():
    """Set up environment variables with default values if not already set."""
    env_defaults = {
        "VOID_DIM": "64",
        "NOISE_DIM": "100",
        "COLOR": "1",
        "INSTANCE": "0",
        "MINIBATCH_SIZE": "32",
        "EPOCHS": "100",
        "MODEL_CHECKPOINT": "5",
        "PICTURE_SNAPSHOT": "1",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_EXPERIMENT": "deepSculpt",
        "MLFLOW_MODEL_NAME": "deepSculpt_generator",
        "PREFECT_FLOW_NAME": "deepSculpt_workflow",
        "PREFECT_BACKEND": "development",
        "DEEPSCULPT_API_URL": "http://localhost:8000"
    }
    
    for key, default in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default
            print(f"Setting default environment variable: {key}={default}")


def train_model(args):
    """
    Train a DeepSculpt model (unified entry point for TensorFlow/PyTorch).
    
    Args:
        args: Command line arguments
    """
    # Route to appropriate training function based on framework
    if hasattr(args, 'framework') and args.framework == "pytorch":
        return train_pytorch_model(args)
    elif hasattr(args, 'framework') and args.framework == "tensorflow":
        return train_tensorflow_model(args)
    else:
        # Default to TensorFlow for backward compatibility
        return train_tensorflow_model(args)


def train_tensorflow_model(args):
    """
    Train a TensorFlow DeepSculpt model (legacy).
    
    Args:
        args: Command line arguments
    """
    if not TF_AVAILABLE:
        print("Error: TensorFlow not available. Please install TensorFlow or use --framework=pytorch")
        return 1
    
    print(f"Starting TensorFlow training with model type: {args.model_type}")
    
    # Set environment variables for compatibility
    os.environ["VOID_DIM"] = str(args.void_dim)
    os.environ["NOISE_DIM"] = str(args.noise_dim)
    os.environ["COLOR"] = "1" if args.color else "0"
    
    # Create results directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    snapshot_dir = os.path.join(results_dir, "snapshots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    print(f"Results will be saved to {results_dir}")
    
    # Create data DataFrame
    print(f"Processing data from folder: {args.data_folder}")
    
    if args.data_file and os.path.exists(args.data_file):
        # Load existing DataFrame if provided
        print(f"Loading data paths from: {args.data_file}")
        data_df = pd.read_csv(args.data_file)
    else:
        # Create new DataFrame from data folder
        print(f"Scanning data folder: {args.data_folder}")
        data_df = create_data_dataframe(args.data_folder)
    
    if data_df.empty:
        print("Error: No data files found! Please check your data folder.")
        return 1
    
    print(f"Found {len(data_df)} data pairs")
    
    # Save DataFrame for future use
    data_file_path = os.path.join(results_dir, "data_paths.csv")
    data_df.to_csv(data_file_path, index=False)
    print(f"Saved data paths to: {data_file_path}")
    
    # Create models
    print(f"Creating TensorFlow {args.model_type} models")
    generator = TFModelFactory.create_generator(
        model_type=args.model_type, 
        void_dim=args.void_dim, 
        noise_dim=args.noise_dim,
        color_mode=1 if args.color else 0
    )
    
    discriminator = TFModelFactory.create_discriminator(
        model_type=args.model_type,
        void_dim=args.void_dim,
        noise_dim=args.noise_dim,
        color_mode=1 if args.color else 0
    )
    
    # Add regularization if requested
    if hasattr(args, 'dropout') and args.dropout > 0:
        from models import add_regularization
        generator = add_regularization(generator, dropout_rate=args.dropout)
        print(f"Added dropout regularization with rate: {args.dropout}")
    
    # Print model summaries
    if args.verbose:
        print("\nGenerator Summary:")
        generator.summary()
        
        print("\nDiscriminator Summary:")
        discriminator.summary()
    
    # Create data loader
    data_loader = DataFrameDataLoader(
        df=data_df,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Create trainer
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs")
    metrics = trainer.train(
        data_loader=data_loader,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        snapshot_dir=snapshot_dir,
        snapshot_freq=args.snapshot_freq
    )
    
    # Save the final models
    generator.save(os.path.join(results_dir, "generator_final"))
    discriminator.save(os.path.join(results_dir, "discriminator_final"))
    
    # Plot and save metrics
    metrics_path = os.path.join(results_dir, "training_metrics.png")
    trainer.plot_metrics(save_path=metrics_path)
    
    # Save to MLflow if requested
    if args.mlflow:
        print("Saving model to MLflow")
        
        # Prepare parameters and metrics
        params = {
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "color_mode": 1 if args.color else 0,
            "dropout": args.dropout
        }
        
        final_metrics = {}
        if metrics.get("gen_loss") and metrics.get("disc_loss"):
            final_metrics = {
                "final_gen_loss": float(metrics["gen_loss"][-1]),
                "final_disc_loss": float(metrics["disc_loss"][-1]),
                "training_time": sum(metrics.get("epoch_times", [0]))
            }
        
        # Save to MLflow
        Manager.save_mlflow_model(
            metrics=final_metrics,
            params=params,
            model=generator
        )
    
    print(f"Training complete! Results saved to {results_dir}")
    return 0


def train_pytorch_model(args):
    """
    Train a PyTorch DeepSculpt model.
    
    Args:
        args: Command line arguments
    """
    if not PYTORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install PyTorch to use this feature.")
        return 1
    
    print(f"Starting PyTorch training with model type: {args.model_type}")
    
    # Create results directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"pytorch_{args.model_type}_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    snapshot_dir = os.path.join(results_dir, "snapshots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    print(f"Results will be saved to {results_dir}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create PyTorch data collector
    print(f"Processing data from folder: {args.data_folder}")
    
    collector_config = {
        "void_dim": args.void_dim,
        "num_shapes": 5,  # Default number of shapes per sculpture
        "sparse_threshold": args.sparse_threshold if args.sparse else 1.0
    }
    
    collector = PyTorchCollector(
        sculptor_config=collector_config,
        output_format="pytorch",
        sparse_threshold=args.sparse_threshold if args.sparse else 1.0,
        device=device
    )
    
    # Create dataset
    if args.data_file and os.path.exists(args.data_file):
        print(f"Loading existing dataset from: {args.data_file}")
        # Load existing dataset paths
        with open(args.data_file, 'r') as f:
            data_paths = json.load(f)
        dataset = collector.load_existing_dataset(data_paths)
    else:
        print(f"Generating new dataset with {args.num_samples} samples")
        dataset = collector.create_streaming_dataset(args.num_samples)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda"
    )
    
    print(f"Created data loader with batch size: {args.batch_size}")
    
    # Create models
    print(f"Creating PyTorch {args.model_type} models")
    generator = PyTorchModelFactory.create_gan_generator(
        model_type=args.model_type,
        void_dim=args.void_dim,
        noise_dim=args.noise_dim,
        color_mode=1 if args.color else 0,
        sparse=args.sparse
    ).to(device)
    
    discriminator = PyTorchModelFactory.create_gan_discriminator(
        model_type=args.model_type,
        void_dim=args.void_dim,
        color_mode=1 if args.color else 0,
        sparse=args.sparse
    ).to(device)
    
    # Print model summaries
    if args.verbose:
        print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create optimizers
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2)
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2)
    )
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        device=device,
        mixed_precision=args.mixed_precision,
        gradient_clip=args.gradient_clip
    )
    
    # Setup MLflow tracking if requested
    mlflow_tracker = None
    if args.mlflow:
        mlflow_tracker = PyTorchMLflowTracker(
            experiment_name=f"pytorch_{args.model_type}",
            run_name=f"{args.model_type}_{timestamp}"
        )
        
        # Log parameters
        params = {
            "framework": "pytorch",
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "color_mode": 1 if args.color else 0,
            "sparse": args.sparse,
            "mixed_precision": args.mixed_precision,
            "device": device
        }
        mlflow_tracker.log_params(params)
    
    # Train the model
    print(f"Starting PyTorch training for {args.epochs} epochs")
    metrics = trainer.train(
        data_loader=data_loader,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        snapshot_dir=snapshot_dir,
        snapshot_freq=args.snapshot_freq,
        mlflow_tracker=mlflow_tracker
    )
    
    # Save the final models
    torch.save(generator.state_dict(), os.path.join(results_dir, "generator_final.pt"))
    torch.save(discriminator.state_dict(), os.path.join(results_dir, "discriminator_final.pt"))
    
    # Save model architectures
    with open(os.path.join(results_dir, "model_config.json"), 'w') as f:
        config = {
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            "color_mode": 1 if args.color else 0,
            "sparse": args.sparse
        }
        json.dump(config, f, indent=2)
    
    # Create visualizations
    if args.generate_samples:
        print("Generating sample visualizations...")
        visualizer = PyTorchVisualizer(device=device)
        
        # Generate samples
        with torch.no_grad():
            noise = torch.randn(4, args.noise_dim, device=device)
            samples = generator(noise)
            
            # Save visualizations
            for i, sample in enumerate(samples):
                vis_path = os.path.join(results_dir, f"sample_{i}.png")
                visualizer.plot_sculpture(sample.cpu(), save_path=vis_path)
    
    # Close MLflow tracker
    if mlflow_tracker:
        mlflow_tracker.end_run()
    
    print(f"PyTorch training complete! Results saved to {results_dir}")
    return 0


def train_diffusion_model(args):
    """
    Train a PyTorch diffusion model.
    
    Args:
        args: Command line arguments
    """
    if not PYTORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install PyTorch to use this feature.")
        return 1
    
    print("Starting PyTorch diffusion model training")
    
    # Create results directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"diffusion_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Results will be saved to {results_dir}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create data collector
    collector_config = {
        "void_dim": args.void_dim,
        "num_shapes": 5,
        "sparse_threshold": args.sparse_threshold if args.sparse else 1.0
    }
    
    collector = PyTorchCollector(
        sculptor_config=collector_config,
        output_format="pytorch",
        sparse_threshold=args.sparse_threshold if args.sparse else 1.0,
        device=device
    )
    
    # Create dataset
    print(f"Generating dataset with {args.num_samples} samples")
    dataset = collector.create_streaming_dataset(args.num_samples)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda"
    )
    
    # Create diffusion model
    print("Creating diffusion model")
    model = PyTorchModelFactory.create_diffusion_model(
        model_type="unet3d",
        void_dim=args.void_dim,
        timesteps=args.timesteps,
        sparse=args.sparse
    ).to(device)
    
    # Create noise scheduler
    noise_scheduler = NoiseScheduler(
        schedule_type=args.noise_schedule,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    # Create diffusion pipeline
    diffusion_pipeline = Diffusion3DPipeline(
        model=model,
        noise_scheduler=noise_scheduler,
        timesteps=args.timesteps
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        diffusion_pipeline=diffusion_pipeline,
        optimizer=optimizer,
        device=device,
        mixed_precision=args.mixed_precision
    )
    
    # Setup MLflow tracking
    mlflow_tracker = None
    if args.mlflow:
        mlflow_tracker = PyTorchMLflowTracker(
            experiment_name="pytorch_diffusion",
            run_name=f"diffusion_{timestamp}"
        )
        
        params = {
            "framework": "pytorch",
            "model_type": "diffusion",
            "void_dim": args.void_dim,
            "timesteps": args.timesteps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "noise_schedule": args.noise_schedule,
            "sparse": args.sparse,
            "mixed_precision": args.mixed_precision,
            "device": device
        }
        mlflow_tracker.log_params(params)
    
    # Train the model
    print(f"Starting diffusion training for {args.epochs} epochs")
    metrics = trainer.train(
        data_loader=data_loader,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        mlflow_tracker=mlflow_tracker
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'noise_scheduler': noise_scheduler,
        'config': {
            'void_dim': args.void_dim,
            'timesteps': args.timesteps,
            'sparse': args.sparse
        }
    }, os.path.join(results_dir, "diffusion_final.pt"))
    
    # Close MLflow tracker
    if mlflow_tracker:
        mlflow_tracker.end_run()
    
    print(f"Diffusion training complete! Results saved to {results_dir}")
    return 0


def sample_diffusion_model(args):
    """
    Generate samples using a trained diffusion model.
    
    Args:
        args: Command line arguments
    """
    if not PYTORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install PyTorch to use this feature.")
        return 1
    
    print(f"Loading diffusion model from: {args.checkpoint}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = PyTorchModelFactory.create_diffusion_model(
        model_type="unet3d",
        void_dim=config['void_dim'],
        timesteps=config.get('timesteps', 1000),
        sparse=config.get('sparse', False)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion pipeline
    noise_scheduler = checkpoint['noise_scheduler']
    diffusion_pipeline = Diffusion3DPipeline(
        model=model,
        noise_scheduler=noise_scheduler,
        timesteps=config.get('timesteps', 1000)
    )
    
    # Create output directory
    output_dir = args.output_dir or "./diffusion_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples...")
    
    # Generate samples
    with torch.no_grad():
        for i in range(args.num_samples):
            print(f"Generating sample {i+1}/{args.num_samples}")
            
            # Sample from diffusion model
            shape = (1, 1 if not config.get('sparse', False) else 2, 
                    config['void_dim'], config['void_dim'], config['void_dim'])
            sample = diffusion_pipeline.sample(
                shape=shape,
                num_steps=args.num_steps,
                device=device
            )
            
            # Save sample
            sample_path = os.path.join(output_dir, f"sample_{i:04d}.pt")
            torch.save(sample.cpu(), sample_path)
            
            # Create visualization if requested
            if args.visualize:
                visualizer = PyTorchVisualizer(device=device)
                vis_path = os.path.join(output_dir, f"sample_{i:04d}.png")
                visualizer.plot_sculpture(sample.squeeze().cpu(), save_path=vis_path)
    
    print(f"Generated {args.num_samples} samples in {output_dir}")
    return 0


def migrate_tensorflow_model(args):
    """
    Migrate a TensorFlow model to PyTorch format.
    
    Args:
        args: Command line arguments
    """
    if not TF_AVAILABLE or not PYTORCH_AVAILABLE:
        print("Error: Both TensorFlow and PyTorch are required for model migration.")
        return 1
    
    print(f"Migrating TensorFlow model from: {args.tf_checkpoint}")
    print(f"Output PyTorch model to: {args.pytorch_output}")
    
    # This is a placeholder for the actual migration logic
    # The full implementation would require detailed weight mapping
    print("Warning: Model migration is not yet fully implemented.")
    print("This would require:")
    print("1. Loading TensorFlow checkpoint")
    print("2. Creating equivalent PyTorch model")
    print("3. Mapping and converting weights")
    print("4. Validating equivalence")
    
    # Create output directory
    os.makedirs(args.pytorch_output, exist_ok=True)
    
    # Save migration info
    migration_info = {
        "source": args.tf_checkpoint,
        "target": args.pytorch_output,
        "timestamp": datetime.now().isoformat(),
        "status": "placeholder_implementation"
    }
    
    with open(os.path.join(args.pytorch_output, "migration_info.json"), 'w') as f:
        json.dump(migration_info, f, indent=2)
    
    print("Migration placeholder completed. Full implementation needed.")
    return 0


def generate_pytorch_data(args):
    """
    Generate PyTorch dataset using the data generation pipeline.
    
    Args:
        args: Command line arguments
    """
    if not PYTORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install PyTorch to use this feature.")
        return 1
    
    print(f"Generating PyTorch dataset with {args.num_samples} samples")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir or "./pytorch_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure sculptor
    sculptor_config = {
        "void_dim": args.void_dim,
        "num_shapes": args.num_shapes,
        "sparse_threshold": args.sparse_threshold if args.sparse else 1.0
    }
    
    # Create collector
    collector = PyTorchCollector(
        sculptor_config=sculptor_config,
        output_format="pytorch",
        sparse_threshold=args.sparse_threshold if args.sparse else 1.0,
        device=device
    )
    
    print(f"Generating {args.num_samples} samples...")
    
    # Generate dataset
    dataset_paths = collector.create_collection(args.num_samples)
    
    # Save dataset metadata
    metadata = {
        "num_samples": args.num_samples,
        "void_dim": args.void_dim,
        "num_shapes": args.num_shapes,
        "sparse": args.sparse,
        "sparse_threshold": args.sparse_threshold if args.sparse else 1.0,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "dataset_paths": dataset_paths
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return 0


def evaluate_pytorch_model(args):
    """
    Evaluate a trained PyTorch model.
    
    Args:
        args: Command line arguments
    """
    if not PYTORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install PyTorch to use this feature.")
        return 1
    
    print(f"Evaluating PyTorch model: {args.checkpoint}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load model configuration
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Warning: Model configuration not found. Using defaults.")
        config = {
            "model_type": "skip",
            "void_dim": 64,
            "noise_dim": 100,
            "color_mode": 1,
            "sparse": False
        }
    
    # Create model
    if args.model_type == "diffusion":
        model = PyTorchModelFactory.create_diffusion_model(
            model_type="unet3d",
            void_dim=config['void_dim'],
            timesteps=1000,
            sparse=config.get('sparse', False)
        ).to(device)
    else:
        model = PyTorchModelFactory.create_gan_generator(
            model_type=config['model_type'],
            void_dim=config['void_dim'],
            noise_dim=config['noise_dim'],
            color_mode=config['color_mode'],
            sparse=config.get('sparse', False)
        ).to(device)
    
    # Load checkpoint
    if args.model_type == "diffusion":
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    model.eval()
    
    # Create output directory
    output_dir = args.output_dir or "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate evaluation samples
    print(f"Generating {args.num_samples} evaluation samples...")
    
    samples = []
    with torch.no_grad():
        for i in range(args.num_samples):
            if args.model_type == "diffusion":
                # For diffusion models, we need the full pipeline
                noise_scheduler = checkpoint.get('noise_scheduler')
                if noise_scheduler:
                    diffusion_pipeline = Diffusion3DPipeline(
                        model=model,
                        noise_scheduler=noise_scheduler,
                        timesteps=1000
                    )
                    shape = (1, 1, config['void_dim'], config['void_dim'], config['void_dim'])
                    sample = diffusion_pipeline.sample(shape=shape, num_steps=50, device=device)
                else:
                    print("Warning: No noise scheduler found. Generating random sample.")
                    sample = torch.randn(1, 1, config['void_dim'], config['void_dim'], config['void_dim'])
            else:
                # For GAN models
                noise = torch.randn(1, config['noise_dim'], device=device)
                sample = model(noise)
            
            samples.append(sample.cpu())
            
            # Save individual sample
            sample_path = os.path.join(output_dir, f"sample_{i:04d}.pt")
            torch.save(sample.cpu(), sample_path)
    
    # Create visualizations if requested
    if args.visualize:
        print("Creating visualizations...")
        visualizer = PyTorchVisualizer(device=device)
        
        for i, sample in enumerate(samples):
            vis_path = os.path.join(output_dir, f"sample_{i:04d}.png")
            visualizer.plot_sculpture(sample.squeeze(), save_path=vis_path)
    
    # Calculate basic statistics
    all_samples = torch.cat(samples, dim=0)
    stats = {
        "num_samples": len(samples),
        "mean_value": float(all_samples.mean()),
        "std_value": float(all_samples.std()),
        "min_value": float(all_samples.min()),
        "max_value": float(all_samples.max()),
        "sparsity": float((all_samples == 0).float().mean()),
        "model_type": args.model_type,
        "checkpoint": args.checkpoint
    }
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Statistics: {stats}")
    
    return 0


def compare_models(args):
    """
    Compare TensorFlow and PyTorch model outputs.
    
    Args:
        args: Command line arguments
    """
    if not (TF_AVAILABLE and PYTORCH_AVAILABLE):
        print("Error: Both TensorFlow and PyTorch are required for model comparison.")
        return 1
    
    print("Comparing TensorFlow and PyTorch models")
    print(f"TensorFlow checkpoint: {args.tf_checkpoint}")
    print(f"PyTorch checkpoint: {args.pytorch_checkpoint}")
    
    # Set device for PyTorch
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir or "./model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a placeholder for the actual comparison logic
    print("Warning: Model comparison is not yet fully implemented.")
    print("This would require:")
    print("1. Loading both TensorFlow and PyTorch models")
    print("2. Generating samples from both models with same inputs")
    print("3. Computing similarity metrics")
    print("4. Creating comparison visualizations")
    
    # Save comparison info
    comparison_info = {
        "tf_checkpoint": args.tf_checkpoint,
        "pytorch_checkpoint": args.pytorch_checkpoint,
        "timestamp": datetime.now().isoformat(),
        "status": "placeholder_implementation",
        "device": device
    }
    
    with open(os.path.join(output_dir, "comparison_info.json"), 'w') as f:
        json.dump(comparison_info, f, indent=2)
    
    print("Comparison placeholder completed. Full implementation needed.")
    return 0


def run_distributed_training(args):
    """
    Run distributed PyTorch training across multiple GPUs.
    
    Args:
        args: Command line arguments
    """
    if not PYTORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install PyTorch to use this feature.")
        return 1
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Distributed training requires GPU support.")
        return 1
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) available. Consider using regular training.")
    
    print(f"Starting distributed training on {num_gpus} GPUs")
    
    # This is a placeholder for distributed training setup
    print("Warning: Distributed training is not yet fully implemented.")
    print("This would require:")
    print("1. Setting up distributed process group")
    print("2. Wrapping models with DistributedDataParallel")
    print("3. Configuring distributed data loading")
    print("4. Handling synchronization and communication")
    
    # For now, fall back to regular training
    print("Falling back to regular PyTorch training...")
    return train_pytorch_model(args)


def run_api_server(args):
    """
    Run the FastAPI server.
    
    Args:
        args: Command line arguments
    """
    print(f"Starting API server on port {args.port}")
    
    # Run with uvicorn
    uvicorn.run(
        "api:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )
    
    return 0


def run_telegram_bot(args):
    """
    Run the Telegram bot.
    
    Args:
        args: Command line arguments
    """
    print("Starting Telegram bot")
    
    # Set Telegram token
    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    elif "TELEGRAM_BOT_TOKEN" not in os.environ:
        print("Error: Telegram bot token not provided. Use --token or set TELEGRAM_BOT_TOKEN environment variable")
        return 1
    
    # Set API URL if provided
    if args.api_url:
        os.environ["DEEPSCULPT_API_URL"] = args.api_url
    
    # Run the bot
    try:
        bot.main()
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error running bot: {e}")
        return 1
    
    return 0


def run_workflow(args):
    """
    Run the DeepSculpt workflow (unified for TensorFlow/PyTorch).
    
    Args:
        args: Command line arguments
    """
    framework = getattr(args, 'framework', 'tensorflow')
    print(f"Running {framework} workflow in {args.mode} mode")
    
    # Set mode
    os.environ["PREFECT_BACKEND"] = args.mode
    
    if framework == "pytorch" and PYTORCH_AVAILABLE:
        return run_pytorch_workflow(args)
    else:
        return run_tensorflow_workflow(args)


def run_pytorch_workflow(args):
    """
    Run the PyTorch-based workflow.
    
    Args:
        args: Command line arguments
    """
    print("Running PyTorch workflow")
    
    # Use PyTorch workflow manager
    pytorch_manager = PyTorchManager()
    
    # Create workflow configuration
    config = {
        "mode": args.mode,
        "data_folder": getattr(args, 'data_folder', './data'),
        "model_type": getattr(args, 'model_type', 'skip'),
        "epochs": getattr(args, 'epochs', 10),
        "framework": "pytorch"
    }
    
    try:
        # Run PyTorch workflow
        result = pytorch_manager.run_full_experiment(config)
        print(f"PyTorch workflow completed successfully: {result}")
        return 0
    except Exception as e:
        print(f"Error running PyTorch workflow: {e}")
        return 1


def run_tensorflow_workflow(args):
    """
    Run the TensorFlow-based workflow (legacy).
    
    Args:
        args: Command line arguments
    """
    print("Running TensorFlow workflow")
    
    # Run the original workflow
    from workflow import main as workflow_main
    
    # Create sys.argv for workflow
    sys_argv = ["workflow.py", f"--mode={args.mode}"]
    if args.data_folder:
        sys_argv.append(f"--data-folder={args.data_folder}")
    if args.model_type:
        sys_argv.append(f"--model-type={args.model_type}")
    if args.epochs:
        sys_argv.append(f"--epochs={args.epochs}")
    if args.schedule:
        sys_argv.append("--schedule")
    
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        # Replace sys.argv
        sys.argv = sys_argv
        
        # Run workflow
        workflow_main()
    except Exception as e:
        print(f"Error running workflow: {e}")
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    return 0


def run_all_services(args):
    """
    Run all services (API server, Telegram bot, and optionally workflow).
    
    Args:
        args: Command line arguments
    """
    print("Starting all DeepSculpt services")
    
    # Define process functions
    def run_api():
        api_args = argparse.Namespace(
            host=args.host,
            port=args.port,
            reload=False
        )
        run_api_server(api_args)
    
    def run_bot():
        bot_args = argparse.Namespace(
            token=args.token,
            api_url=f"http://{args.host}:{args.port}"
        )
        run_telegram_bot(bot_args)
    
    def run_work():
        workflow_args = argparse.Namespace(
            framework=getattr(args, 'framework', 'tensorflow'),
            mode=args.mode,
            data_folder=args.data_folder,
            model_type=args.model_type,
            epochs=args.epochs,
            schedule=args.schedule
        )
        run_workflow(workflow_args)
    
    # Create and start processes
    processes = []
    
    # API process
    api_process = multiprocessing.Process(target=run_api)
    api_process.start()
    processes.append(api_process)
    print(f"API server started on {args.host}:{args.port}")
    
    # Give API time to start
    time.sleep(3)
    
    # Bot process
    bot_process = multiprocessing.Process(target=run_bot)
    bot_process.start()
    processes.append(bot_process)
    print("Telegram bot started")
    
    # Workflow process (if requested)
    if args.workflow:
        workflow_process = multiprocessing.Process(target=run_work)
        workflow_process.start()
        processes.append(workflow_process)
        print(f"Workflow started in {args.mode} mode")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("Shutting down all services...")
        for process in processes:
            if process.is_alive():
                process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for processes
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
        for process in processes:
            if process.is_alive():
                process.terminate()
    
    return 0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSculpt - Unified Deep Learning for 3D Generation")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command (unified TensorFlow/PyTorch)
    train_parser = subparsers.add_parser("train", help="Train a model (GAN)")
    train_parser.add_argument("--framework", type=str, default="tensorflow",
                            choices=["tensorflow", "pytorch"],
                            help="Framework to use (tensorflow or pytorch)")
    train_parser.add_argument("--model-type", type=str, default="skip",
                            choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                            help="Type of model to train")
    train_parser.add_argument("--epochs", type=int, default=100,
                            help="Number of epochs to train for")
    train_parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, default=0.0002,
                            help="Learning rate for optimizers")
    train_parser.add_argument("--beta1", type=float, default=0.5,
                            help="Beta1 parameter for Adam optimizer")
    train_parser.add_argument("--beta2", type=float, default=0.999,
                            help="Beta2 parameter for Adam optimizer")
    train_parser.add_argument("--void-dim", type=int, default=64,
                            help="Dimension of void space")
    train_parser.add_argument("--noise-dim", type=int, default=100,
                            help="Dimension of noise vector")
    train_parser.add_argument("--color", action="store_true", default=True,
                            help="Use color mode")
    train_parser.add_argument("--snapshot-freq", type=int, default=5,
                            help="Frequency of saving snapshots (epochs)")
    train_parser.add_argument("--data-folder", type=str, default="./data",
                            help="Path to data folder")
    train_parser.add_argument("--data-file", type=str, default=None,
                            help="Path to data file (CSV for TF, JSON for PyTorch)")
    train_parser.add_argument("--output-dir", type=str, default="./results",
                            help="Directory for output files")
    train_parser.add_argument("--dropout", type=float, default=0.0,
                            help="Dropout rate for regularization (TensorFlow only)")
    train_parser.add_argument("--mlflow", action="store_true",
                            help="Save model to MLflow")
    train_parser.add_argument("--verbose", action="store_true",
                            help="Print verbose output")
    
    # PyTorch-specific arguments
    train_parser.add_argument("--sparse", action="store_true",
                            help="Use sparse tensors (PyTorch only)")
    train_parser.add_argument("--sparse-threshold", type=float, default=0.1,
                            help="Sparsity threshold for sparse tensor conversion")
    train_parser.add_argument("--mixed-precision", action="store_true", default=True,
                            help="Use mixed precision training (PyTorch only)")
    train_parser.add_argument("--gradient-clip", type=float, default=1.0,
                            help="Gradient clipping value (PyTorch only)")
    train_parser.add_argument("--num-workers", type=int, default=4,
                            help="Number of data loader workers (PyTorch only)")
    train_parser.add_argument("--num-samples", type=int, default=1000,
                            help="Number of samples to generate for training (PyTorch only)")
    train_parser.add_argument("--cpu", action="store_true",
                            help="Force CPU usage (PyTorch only)")
    train_parser.add_argument("--generate-samples", action="store_true",
                            help="Generate sample visualizations after training")
    
    # Diffusion training command
    diffusion_parser = subparsers.add_parser("train-diffusion", help="Train a diffusion model (PyTorch only)")
    diffusion_parser.add_argument("--epochs", type=int, default=100,
                                help="Number of epochs to train for")
    diffusion_parser.add_argument("--batch-size", type=int, default=16,
                                help="Batch size for training")
    diffusion_parser.add_argument("--learning-rate", type=float, default=1e-4,
                                help="Learning rate for optimizer")
    diffusion_parser.add_argument("--weight-decay", type=float, default=1e-4,
                                help="Weight decay for optimizer")
    diffusion_parser.add_argument("--void-dim", type=int, default=64,
                                help="Dimension of void space")
    diffusion_parser.add_argument("--timesteps", type=int, default=1000,
                                help="Number of diffusion timesteps")
    diffusion_parser.add_argument("--noise-schedule", type=str, default="linear",
                                choices=["linear", "cosine", "sigmoid"],
                                help="Noise scheduling strategy")
    diffusion_parser.add_argument("--beta-start", type=float, default=0.0001,
                                help="Starting beta value for noise schedule")
    diffusion_parser.add_argument("--beta-end", type=float, default=0.02,
                                help="Ending beta value for noise schedule")
    diffusion_parser.add_argument("--data-folder", type=str, default="./data",
                                help="Path to data folder")
    diffusion_parser.add_argument("--output-dir", type=str, default="./results",
                                help="Directory for output files")
    diffusion_parser.add_argument("--sparse", action="store_true",
                                help="Use sparse tensors")
    diffusion_parser.add_argument("--sparse-threshold", type=float, default=0.1,
                                help="Sparsity threshold for sparse tensor conversion")
    diffusion_parser.add_argument("--mixed-precision", action="store_true", default=True,
                                help="Use mixed precision training")
    diffusion_parser.add_argument("--num-workers", type=int, default=4,
                                help="Number of data loader workers")
    diffusion_parser.add_argument("--num-samples", type=int, default=1000,
                                help="Number of samples to generate for training")
    diffusion_parser.add_argument("--cpu", action="store_true",
                                help="Force CPU usage")
    diffusion_parser.add_argument("--mlflow", action="store_true",
                                help="Save model to MLflow")
    
    # Diffusion sampling command
    sample_parser = subparsers.add_parser("sample-diffusion", help="Generate samples from diffusion model")
    sample_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Path to diffusion model checkpoint")
    sample_parser.add_argument("--num-samples", type=int, default=10,
                             help="Number of samples to generate")
    sample_parser.add_argument("--num-steps", type=int, default=50,
                             help="Number of denoising steps")
    sample_parser.add_argument("--output-dir", type=str, default=None,
                             help="Output directory for samples")
    sample_parser.add_argument("--visualize", action="store_true",
                             help="Create visualizations of samples")
    sample_parser.add_argument("--cpu", action="store_true",
                             help="Force CPU usage")
    
    # Model migration command
    migrate_parser = subparsers.add_parser("migrate-model", help="Migrate TensorFlow model to PyTorch")
    migrate_parser.add_argument("--tf-checkpoint", type=str, required=True,
                              help="Path to TensorFlow checkpoint")
    migrate_parser.add_argument("--pytorch-output", type=str, required=True,
                              help="Output directory for PyTorch model")
    migrate_parser.add_argument("--model-type", type=str, default="skip",
                              choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                              help="Type of model to migrate")
    migrate_parser.add_argument("--validate", action="store_true",
                              help="Validate migration by comparing outputs")
    
    # Data generation command
    data_parser = subparsers.add_parser("generate-data", help="Generate PyTorch dataset")
    data_parser.add_argument("--num-samples", type=int, default=1000,
                           help="Number of samples to generate")
    data_parser.add_argument("--void-dim", type=int, default=64,
                           help="Dimension of void space")
    data_parser.add_argument("--num-shapes", type=int, default=5,
                           help="Number of shapes per sculpture")
    data_parser.add_argument("--output-dir", type=str, default=None,
                           help="Output directory for dataset")
    data_parser.add_argument("--sparse", action="store_true",
                           help="Use sparse tensors")
    data_parser.add_argument("--sparse-threshold", type=float, default=0.1,
                           help="Sparsity threshold for sparse tensor conversion")
    data_parser.add_argument("--cpu", action="store_true",
                           help="Force CPU usage")
    
    # Model evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Path to model checkpoint")
    eval_parser.add_argument("--model-type", type=str, default="gan",
                           choices=["gan", "diffusion"],
                           help="Type of model to evaluate")
    eval_parser.add_argument("--num-samples", type=int, default=10,
                           help="Number of samples to generate for evaluation")
    eval_parser.add_argument("--output-dir", type=str, default=None,
                           help="Output directory for evaluation results")
    eval_parser.add_argument("--visualize", action="store_true",
                           help="Create visualizations of samples")
    eval_parser.add_argument("--cpu", action="store_true",
                           help="Force CPU usage")
    
    # Model comparison command
    compare_parser = subparsers.add_parser("compare-models", help="Compare TensorFlow and PyTorch models")
    compare_parser.add_argument("--tf-checkpoint", type=str, required=True,
                              help="Path to TensorFlow checkpoint")
    compare_parser.add_argument("--pytorch-checkpoint", type=str, required=True,
                              help="Path to PyTorch checkpoint")
    compare_parser.add_argument("--num-samples", type=int, default=10,
                              help="Number of samples to generate for comparison")
    compare_parser.add_argument("--output-dir", type=str, default=None,
                              help="Output directory for comparison results")
    compare_parser.add_argument("--cpu", action="store_true",
                              help="Force CPU usage")
    
    # Distributed training command
    distributed_parser = subparsers.add_parser("train-distributed", help="Run distributed PyTorch training")
    distributed_parser.add_argument("--model-type", type=str, default="skip",
                                   choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                                   help="Type of model to train")
    distributed_parser.add_argument("--epochs", type=int, default=100,
                                   help="Number of epochs to train for")
    distributed_parser.add_argument("--batch-size", type=int, default=32,
                                   help="Batch size per GPU")
    distributed_parser.add_argument("--learning-rate", type=float, default=0.0002,
                                   help="Learning rate for optimizers")
    distributed_parser.add_argument("--void-dim", type=int, default=64,
                                   help="Dimension of void space")
    distributed_parser.add_argument("--noise-dim", type=int, default=100,
                                   help="Dimension of noise vector")
    distributed_parser.add_argument("--data-folder", type=str, default="./data",
                                   help="Path to data folder")
    distributed_parser.add_argument("--output-dir", type=str, default="./results",
                                   help="Directory for output files")
    distributed_parser.add_argument("--num-samples", type=int, default=1000,
                                   help="Number of samples to generate for training")
    distributed_parser.add_argument("--sparse", action="store_true",
                                   help="Use sparse tensors")
    distributed_parser.add_argument("--mixed-precision", action="store_true", default=True,
                                   help="Use mixed precision training")
    
    # API server command
    api_parser = subparsers.add_parser("serve-api", help="Run the API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0",
                          help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000,
                          help="Port to bind the server to")
    api_parser.add_argument("--reload", action="store_true",
                          help="Enable auto-reload on code changes")
    
    # Telegram bot command
    bot_parser = subparsers.add_parser("run-bot", help="Run the Telegram bot")
    bot_parser.add_argument("--token", type=str, default=None,
                          help="Telegram bot token")
    bot_parser.add_argument("--api-url", type=str, default=None,
                          help="URL of the DeepSculpt API server")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run the workflow")
    workflow_parser.add_argument("--framework", type=str, default="tensorflow",
                               choices=["tensorflow", "pytorch"],
                               help="Framework to use")
    workflow_parser.add_argument("--mode", type=str, choices=["development", "production"],
                               default="development", help="Execution mode")
    workflow_parser.add_argument("--data-folder", type=str, default="./data",
                               help="Path to data folder")
    workflow_parser.add_argument("--model-type", type=str, default="skip",
                               choices=["simple", "complex", "skip", "monochrome"],
                               help="Type of model to train")
    workflow_parser.add_argument("--epochs", type=int, default=10,
                               help="Number of epochs for training")
    workflow_parser.add_argument("--schedule", action="store_true",
                               help="Run with schedule")
    
    # Run all services command
    all_parser = subparsers.add_parser("all", help="Run all services")
    all_parser.add_argument("--framework", type=str, default="tensorflow",
                          choices=["tensorflow", "pytorch"],
                          help="Framework to use for workflow")
    all_parser.add_argument("--host", type=str, default="0.0.0.0",
                          help="Host for API server")
    all_parser.add_argument("--port", type=int, default=8000,
                          help="Port for API server")
    all_parser.add_argument("--token", type=str, default=None,
                          help="Telegram bot token")
    all_parser.add_argument("--mode", type=str, choices=["development", "production"],
                          default="development", help="Execution mode for workflow")
    all_parser.add_argument("--workflow", action="store_true",
                          help="Run workflow in addition to API and bot")
    all_parser.add_argument("--data-folder", type=str, default="./data",
                          help="Path to data folder for workflow")
    all_parser.add_argument("--model-type", type=str, default="skip",
                          choices=["simple", "complex", "skip", "monochrome"],
                          help="Type of model for workflow")
    all_parser.add_argument("--epochs", type=int, default=10,
                          help="Number of epochs for workflow training")
    all_parser.add_argument("--schedule", action="store_true",
                          help="Run workflow with schedule")
    
    return parser.parse_args()


def main():
    """Main entry point for DeepSculpt."""
    # Setup environment
    setup_environment()
    
    # Parse arguments
    args = parse_arguments()
    
    # Check framework availability
    if hasattr(args, 'framework'):
        if args.framework == "pytorch" and not PYTORCH_AVAILABLE:
            print("Error: PyTorch not available. Please install PyTorch or use --framework=tensorflow")
            return 1
        elif args.framework == "tensorflow" and not TF_AVAILABLE:
            print("Error: TensorFlow not available. Please install TensorFlow or use --framework=pytorch")
            return 1
    
    # Execute requested command
    if args.command == "train":
        return train_model(args)
    elif args.command == "train-diffusion":
        return train_diffusion_model(args)
    elif args.command == "sample-diffusion":
        return sample_diffusion_model(args)
    elif args.command == "migrate-model":
        return migrate_tensorflow_model(args)
    elif args.command == "generate-data":
        return generate_pytorch_data(args)
    elif args.command == "evaluate":
        return evaluate_pytorch_model(args)
    elif args.command == "compare-models":
        return compare_models(args)
    elif args.command == "train-distributed":
        return run_distributed_training(args)
    elif args.command == "serve-api":
        return run_api_server(args)
    elif args.command == "run-bot":
        return run_telegram_bot(args)
    elif args.command == "workflow":
        return run_workflow(args)
    elif args.command == "all":
        return run_all_services(args)
    else:
        print("Error: No command specified")
        print("Available commands:")
        print("  train              - Train a GAN model (TensorFlow/PyTorch)")
        print("  train-diffusion    - Train a diffusion model (PyTorch only)")
        print("  sample-diffusion   - Generate samples from diffusion model")
        print("  migrate-model      - Migrate TensorFlow model to PyTorch")
        print("  generate-data      - Generate PyTorch dataset")
        print("  evaluate           - Evaluate a trained model")
        print("  compare-models     - Compare TensorFlow and PyTorch models")
        print("  train-distributed  - Run distributed PyTorch training")
        print("  serve-api          - Run the API server")
        print("  run-bot            - Run the Telegram bot")
        print("  workflow           - Run the workflow")
        print("  all                - Run all services")
        return 1


if __name__ == "__main__":
    sys.exit(main())