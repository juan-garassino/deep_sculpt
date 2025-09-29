#!/usr/bin/env python3
"""
DeepSculpt Legacy - TensorFlow Main Entry Point

This is the legacy main entry point for the DeepSculpt project using TensorFlow.
Provides backward compatibility with existing TensorFlow workflows while maintaining
the original functionality and interface.

Usage:
    # TensorFlow GAN training
    python main.py train --model-type=skip --epochs=100 --data-folder=./data
    
    # Data generation
    python main.py generate-data --num-samples=1000 --output-dir=./data
    
    # Model evaluation
    python main.py evaluate --checkpoint=./models/generator.h5 --num-samples=10
    
    # API and bot services
    python main.py serve-api --port=8000
    python main.py run-bot --token=YOUR_TELEGRAM_TOKEN
    
    # Workflow automation
    python main.py workflow --mode=development
    python main.py all --mode=production
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

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Framework imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} available")
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not available. Please install TensorFlow.")
    sys.exit(1)

# Import DeepSculpt legacy modules
try:
    from deepSculpt.models import ModelFactory as TFModelFactory
    from deepSculpt.trainer import DeepSculptTrainer, DataFrameDataLoader, create_data_dataframe
    from deepSculpt.workflow import Manager, build_flow
    from deepSculpt.collector import Collector
    from deepSculpt.curator import OneHotEncoderDecoder, BinaryEncoderDecoder, RGBEncoderDecoder
    from deepSculpt.visualization import Visualizer
    from deepSculpt.logger import Logger
    from boilerplate import api
    from boilerplate import bot
    
except ImportError as e:
    print(f"Error importing DeepSculpt legacy modules: {e}")
    print("Make sure all required modules are available in the legacy directory")
    sys.exit(1)


class LegacyConfig:
    """Configuration management for legacy TensorFlow operations."""
    
    def __init__(self):
        self.setup_environment()
    
    def setup_environment(self):
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
            "MLFLOW_EXPERIMENT": "deepSculpt_legacy",
            "MLFLOW_MODEL_NAME": "deepSculpt_generator_tf",
            "PREFECT_FLOW_NAME": "deepSculpt_workflow_legacy",
            "PREFECT_BACKEND": "development",
            "DEEPSCULPT_API_URL": "http://localhost:8000"
        }
        
        for key, default in env_defaults.items():
            if key not in os.environ:
                os.environ[key] = default
                print(f"Setting default environment variable: {key}={default}")


def train_tensorflow_model(args):
    """
    Train a TensorFlow DeepSculpt model (legacy implementation).
    
    Args:
        args: Command line arguments
    """
    print(f"Starting TensorFlow training with model type: {args.model_type}")
    
    # Set environment variables for compatibility
    os.environ["VOID_DIM"] = str(args.void_dim)
    os.environ["NOISE_DIM"] = str(args.noise_dim)
    os.environ["COLOR"] = "1" if args.color else "0"
    
    # Create results directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"tf_{args.model_type}_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    snapshot_dir = os.path.join(results_dir, "snapshots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    print(f"Results will be saved to {results_dir}")
    
    # Create data DataFrame
    print(f"Processing data from folder: {args.data_folder}")
    
    if args.data_file and os.path.exists(args.data_file):
        print(f"Loading data paths from: {args.data_file}")
        data_df = pd.read_csv(args.data_file)
    else:
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
        
        params = {
            "framework": "tensorflow",
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "color_mode": 1 if args.color else 0
        }
        
        final_metrics = {}
        if metrics.get("gen_loss") and metrics.get("disc_loss"):
            final_metrics = {
                "final_gen_loss": float(metrics["gen_loss"][-1]),
                "final_disc_loss": float(metrics["disc_loss"][-1]),
                "training_time": sum(metrics.get("epoch_times", [0]))
            }
        
        Manager.save_mlflow_model(
            metrics=final_metrics,
            params=params,
            model=generator
        )
    
    print(f"Training complete! Results saved to {results_dir}")
    return 0


def generate_tensorflow_data(args):
    """
    Generate TensorFlow dataset using the legacy data generation pipeline.
    
    Args:
        args: Command line arguments
    """
    print(f"Generating TensorFlow dataset with {args.num_samples} samples")
    
    # Create output directory
    output_dir = args.output_dir or "./tf_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure collector
    collector = Collector(
        void_dim=args.void_dim,
        num_shapes=args.num_shapes,
        output_dir=output_dir
    )
    
    print(f"Generating {args.num_samples} samples...")
    
    # Generate dataset
    dataset_paths = collector.create_collection(args.num_samples)
    
    # Save dataset metadata
    metadata = {
        "num_samples": args.num_samples,
        "void_dim": args.void_dim,
        "num_shapes": args.num_shapes,
        "framework": "tensorflow",
        "timestamp": datetime.now().isoformat(),
        "dataset_paths": dataset_paths
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"TensorFlow dataset generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return 0


def evaluate_tensorflow_model(args):
    """
    Evaluate a trained TensorFlow model.
    
    Args:
        args: Command line arguments
    """
    print(f"Evaluating TensorFlow model: {args.checkpoint}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(args.checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Create output directory
    output_dir = args.output_dir or "./tf_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate evaluation samples
    print(f"Generating {args.num_samples} evaluation samples...")
    
    samples = []
    for i in range(args.num_samples):
        # Generate random noise
        noise = tf.random.normal([1, args.noise_dim])
        sample = model(noise, training=False)
        samples.append(sample.numpy())
        
        # Save individual sample
        sample_path = os.path.join(output_dir, f"sample_{i:04d}.npy")
        np.save(sample_path, sample.numpy())
    
    # Create visualizations if requested
    if args.visualize:
        print("Creating visualizations...")
        visualizer = Visualizer()
        
        for i, sample in enumerate(samples):
            vis_path = os.path.join(output_dir, f"sample_{i:04d}.png")
            visualizer.plot_sculpture(sample.squeeze(), save_path=vis_path)
    
    # Calculate basic statistics
    all_samples = np.concatenate(samples, axis=0)
    stats = {
        "num_samples": len(samples),
        "mean_value": float(np.mean(all_samples)),
        "std_value": float(np.std(all_samples)),
        "min_value": float(np.min(all_samples)),
        "max_value": float(np.max(all_samples)),
        "sparsity": float(np.mean(all_samples == 0)),
        "framework": "tensorflow",
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


def serve_api(args):
    """
    Start the DeepSculpt API server.
    
    Args:
        args: Command line arguments
    """
    print(f"Starting DeepSculpt API server on port {args.port}")
    
    # Configure API
    api_config = {
        "host": args.host,
        "port": args.port,
        "debug": args.debug,
        "reload": args.reload
    }
    
    try:
        uvicorn.run(
            "boilerplate.api:app",
            host=args.host,
            port=args.port,
            debug=args.debug,
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\nAPI server stopped")
    except Exception as e:
        print(f"Error starting API server: {e}")
        return 1
    
    return 0


def run_bot(args):
    """
    Start the Telegram bot.
    
    Args:
        args: Command line arguments
    """
    print("Starting DeepSculpt Telegram bot")
    
    if not args.token:
        print("Error: Telegram bot token is required")
        return 1
    
    try:
        bot.run_bot(token=args.token)
    except KeyboardInterrupt:
        print("\nBot stopped")
    except Exception as e:
        print(f"Error running bot: {e}")
        return 1
    
    return 0


def run_workflow(args):
    """
    Run the DeepSculpt workflow.
    
    Args:
        args: Command line arguments
    """
    print(f"Running DeepSculpt workflow in {args.mode} mode")
    
    try:
        # Build and run workflow
        flow = build_flow(mode=args.mode)
        result = flow.run()
        
        print(f"Workflow completed successfully: {result}")
        return 0
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        return 1


def run_all_services(args):
    """
    Run all DeepSculpt services (API, bot, workflow).
    
    Args:
        args: Command line arguments
    """
    print(f"Starting all DeepSculpt services in {args.mode} mode")
    
    processes = []
    
    try:
        # Start API server
        api_process = multiprocessing.Process(
            target=serve_api,
            args=(argparse.Namespace(
                host=args.host,
                port=args.port,
                debug=args.debug,
                reload=False
            ),)
        )
        api_process.start()
        processes.append(api_process)
        print(f"API server started on {args.host}:{args.port}")
        
        # Start bot if token provided
        if args.bot_token:
            bot_process = multiprocessing.Process(
                target=run_bot,
                args=(argparse.Namespace(token=args.bot_token),)
            )
            bot_process.start()
            processes.append(bot_process)
            print("Telegram bot started")
        
        # Run workflow
        workflow_process = multiprocessing.Process(
            target=run_workflow,
            args=(argparse.Namespace(mode=args.mode),)
        )
        workflow_process.start()
        processes.append(workflow_process)
        print("Workflow started")
        
        # Wait for processes
        print("All services started. Press Ctrl+C to stop.")
        
        def signal_handler(sig, frame):
            print("\nStopping all services...")
            for process in processes:
                process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Keep main process alive
        while True:
            time.sleep(1)
            # Check if any process died
            for process in processes:
                if not process.is_alive():
                    print(f"Process {process.name} died, restarting...")
                    process.start()
    
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for process in processes:
            process.terminate()
    except Exception as e:
        print(f"Error running services: {e}")
        return 1
    
    return 0


def create_parser():
    """Create the argument parser for the legacy TensorFlow interface."""
    parser = argparse.ArgumentParser(
        description="DeepSculpt Legacy - TensorFlow Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a TensorFlow GAN model
  python main.py train --model-type=skip --epochs=100 --data-folder=./data
  
  # Generate training data
  python main.py generate-data --num-samples=1000 --output-dir=./data
  
  # Evaluate a trained model
  python main.py evaluate --checkpoint=./models/generator.h5 --num-samples=10
  
  # Start API server
  python main.py serve-api --port=8000
  
  # Run Telegram bot
  python main.py run-bot --token=YOUR_TELEGRAM_TOKEN
  
  # Run workflow automation
  python main.py workflow --mode=development
  
  # Run all services
  python main.py all --mode=production --bot-token=YOUR_TOKEN
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a TensorFlow GAN model')
    train_parser.add_argument('--model-type', type=str, default='skip',
                             choices=['simple', 'complex', 'skip', 'monochrome', 'autoencoder'],
                             help='Type of GAN model to train')
    train_parser.add_argument('--void-dim', type=int, default=64,
                             help='Dimension of the 3D voxel space')
    train_parser.add_argument('--noise-dim', type=int, default=100,
                             help='Dimension of the noise vector')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.0002,
                             help='Learning rate for optimizers')
    train_parser.add_argument('--beta1', type=float, default=0.5,
                             help='Beta1 parameter for Adam optimizer')
    train_parser.add_argument('--beta2', type=float, default=0.999,
                             help='Beta2 parameter for Adam optimizer')
    train_parser.add_argument('--data-folder', type=str, default='./data',
                             help='Folder containing training data')
    train_parser.add_argument('--data-file', type=str,
                             help='CSV file with data paths (optional)')
    train_parser.add_argument('--output-dir', type=str, default='./results',
                             help='Directory to save results')
    train_parser.add_argument('--snapshot-freq', type=int, default=10,
                             help='Frequency of saving snapshots (epochs)')
    train_parser.add_argument('--color', action='store_true',
                             help='Enable color mode (RGB)')
    train_parser.add_argument('--mlflow', action='store_true',
                             help='Enable MLflow experiment tracking')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Enable verbose output')
    
    # Data generation command
    data_parser = subparsers.add_parser('generate-data', help='Generate training data')
    data_parser.add_argument('--num-samples', type=int, default=1000,
                            help='Number of samples to generate')
    data_parser.add_argument('--void-dim', type=int, default=64,
                            help='Dimension of the 3D voxel space')
    data_parser.add_argument('--num-shapes', type=int, default=5,
                            help='Number of shapes per sculpture')
    data_parser.add_argument('--output-dir', type=str, default='./data',
                            help='Output directory for generated data')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--num-samples', type=int, default=10,
                            help='Number of samples to generate for evaluation')
    eval_parser.add_argument('--noise-dim', type=int, default=100,
                            help='Dimension of the noise vector')
    eval_parser.add_argument('--output-dir', type=str,
                            help='Output directory for evaluation results')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Create visualizations of generated samples')
    
    # API server command
    api_parser = subparsers.add_parser('serve-api', help='Start the API server')
    api_parser.add_argument('--host', type=str, default='0.0.0.0',
                           help='Host address for the API server')
    api_parser.add_argument('--port', type=int, default=8000,
                           help='Port for the API server')
    api_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
    api_parser.add_argument('--reload', action='store_true',
                           help='Enable auto-reload on code changes')
    
    # Bot command
    bot_parser = subparsers.add_parser('run-bot', help='Start the Telegram bot')
    bot_parser.add_argument('--token', type=str, required=True,
                           help='Telegram bot token')
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run workflow automation')
    workflow_parser.add_argument('--mode', type=str, default='development',
                                choices=['development', 'production'],
                                help='Workflow execution mode')
    
    # All services command
    all_parser = subparsers.add_parser('all', help='Run all services')
    all_parser.add_argument('--mode', type=str, default='development',
                           choices=['development', 'production'],
                           help='Execution mode for all services')
    all_parser.add_argument('--host', type=str, default='0.0.0.0',
                           help='Host address for the API server')
    all_parser.add_argument('--port', type=int, default=8000,
                           help='Port for the API server')
    all_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
    all_parser.add_argument('--bot-token', type=str,
                           help='Telegram bot token (optional)')
    
    return parser


def main():
    """Main entry point for the legacy TensorFlow interface."""
    # Initialize configuration
    config = LegacyConfig()
    
    # Create parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    print(f"DeepSculpt Legacy (TensorFlow) - Command: {args.command}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Route to appropriate function
    try:
        if args.command == 'train':
            return train_tensorflow_model(args)
        elif args.command == 'generate-data':
            return generate_tensorflow_data(args)
        elif args.command == 'evaluate':
            return evaluate_tensorflow_model(args)
        elif args.command == 'serve-api':
            return serve_api(args)
        elif args.command == 'run-bot':
            return run_bot(args)
        elif args.command == 'workflow':
            return run_workflow(args)
        elif args.command == 'all':
            return run_all_services(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error executing command '{args.command}': {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())