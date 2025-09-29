#!/usr/bin/env python3
"""
DeepSculpt v2.0 - Complete Pipeline Example

This example demonstrates how to run the complete DeepSculpt pipeline
from data generation to model training and evaluation.

Usage:
    python complete_pipeline_example.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import DeepSculptPipeline


def run_quick_demo():
    """Run a quick demo of the complete pipeline."""
    print("🚀 DeepSculpt v2.0 - Quick Demo Pipeline")
    print("=" * 50)
    
    # Configuration for quick demo
    config = {
        # Model configuration
        'model_type': 'gan',
        'gan_model_type': 'simple',
        'void_dim': 32,  # Small for quick demo
        'noise_dim': 64,
        
        # Training configuration
        'epochs': 3,  # Very short training
        'batch_size': 4,
        'learning_rate': 0.0002,
        'mixed_precision': True,
        
        # Data configuration
        'num_samples': 20,  # Small dataset
        'num_shapes': 3,
        'sparse_mode': True,
        'sparse_threshold': 0.1,
        
        # Pipeline configuration
        'output_dir': './demo_results',
        'log_level': 'INFO',
        'enable_monitoring': True,
        'enable_optimization': True,
        
        # Evaluation configuration
        'num_eval_samples': 3,
        'viz_backend': 'matplotlib'
    }
    
    # Create and run pipeline
    pipeline = DeepSculptPipeline(config)
    
    print("🎯 Running quick demo pipeline...")
    print("This will:")
    print("  1. Generate 20 synthetic 3D sculptures")
    print("  2. Train a simple GAN for 3 epochs")
    print("  3. Generate 3 sample sculptures")
    print("  4. Evaluate the model")
    print("  5. Create visualizations")
    print()
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print(f"📁 Check results in: {pipeline.base_dir}")
        print("\nGenerated files:")
        print(f"  - Data: {len(list(pipeline.data_dir.glob('*.pt')))} files")
        print(f"  - Models: {len(list(pipeline.models_dir.glob('*.pt')))} files")
        print(f"  - Samples: {len(list(pipeline.samples_dir.glob('*.pt')))} files")
        print(f"  - Visualizations: {len(list(pipeline.visualizations_dir.glob('*.png')))} files")
    else:
        print("\n❌ Demo failed!")
    
    return success


def run_full_pipeline():
    """Run the full pipeline with realistic settings."""
    print("🚀 DeepSculpt v2.0 - Full Pipeline")
    print("=" * 50)
    
    # Configuration for full pipeline
    config = {
        # Model configuration
        'model_type': 'gan',
        'gan_model_type': 'skip',
        'void_dim': 64,
        'noise_dim': 100,
        
        # Training configuration
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0002,
        'mixed_precision': True,
        
        # Data configuration
        'num_samples': 1000,
        'num_shapes': 5,
        'sparse_mode': True,
        'sparse_threshold': 0.1,
        
        # Pipeline configuration
        'output_dir': './full_results',
        'log_level': 'INFO',
        'enable_monitoring': True,
        'enable_optimization': True,
        
        # Evaluation configuration
        'num_eval_samples': 10,
        'viz_backend': 'plotly'
    }
    
    # Create and run pipeline
    pipeline = DeepSculptPipeline(config)
    
    print("🎯 Running full pipeline...")
    print("This will:")
    print("  1. Generate 1000 synthetic 3D sculptures")
    print("  2. Train a skip-connection GAN for 50 epochs")
    print("  3. Generate 10 sample sculptures")
    print("  4. Evaluate the model comprehensively")
    print("  5. Create detailed visualizations")
    print("\n⏰ This may take 30-60 minutes depending on your hardware...")
    print()
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Full pipeline completed successfully!")
        print(f"📁 Check results in: {pipeline.base_dir}")
    else:
        print("\n❌ Full pipeline failed!")
    
    return success


def run_diffusion_pipeline():
    """Run pipeline with diffusion model."""
    print("🚀 DeepSculpt v2.0 - Diffusion Pipeline")
    print("=" * 50)
    
    # Configuration for diffusion pipeline
    config = {
        # Model configuration
        'model_type': 'diffusion',
        'void_dim': 48,  # Slightly smaller for diffusion
        'timesteps': 100,  # Reduced for faster training
        
        # Training configuration
        'epochs': 20,
        'batch_size': 16,  # Smaller batch for diffusion
        'learning_rate': 1e-4,
        'mixed_precision': True,
        'noise_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        
        # Data configuration
        'num_samples': 500,
        'num_shapes': 4,
        'sparse_mode': True,
        'sparse_threshold': 0.1,
        
        # Pipeline configuration
        'output_dir': './diffusion_results',
        'log_level': 'INFO',
        'enable_monitoring': True,
        'enable_optimization': True,
        
        # Evaluation configuration
        'num_eval_samples': 5,
        'diffusion_steps': 20,  # Faster sampling
        'viz_backend': 'plotly'
    }
    
    # Create and run pipeline
    pipeline = DeepSculptPipeline(config)
    
    print("🎯 Running diffusion pipeline...")
    print("This will:")
    print("  1. Generate 500 synthetic 3D sculptures")
    print("  2. Train a 3D U-Net diffusion model for 20 epochs")
    print("  3. Generate 5 sample sculptures using diffusion")
    print("  4. Evaluate the diffusion model")
    print("  5. Create visualizations")
    print("\n⏰ This may take 45-90 minutes depending on your hardware...")
    print()
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Diffusion pipeline completed successfully!")
        print(f"📁 Check results in: {pipeline.base_dir}")
    else:
        print("\n❌ Diffusion pipeline failed!")
    
    return success


def main():
    """Main function to run pipeline examples."""
    print("🎨 DeepSculpt v2.0 - Pipeline Examples")
    print("=" * 60)
    print()
    print("Choose a pipeline to run:")
    print("1. Quick Demo (5-10 minutes)")
    print("2. Full GAN Pipeline (30-60 minutes)")
    print("3. Diffusion Pipeline (45-90 minutes)")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n" + "="*60)
                success = run_quick_demo()
                break
            elif choice == '2':
                print("\n" + "="*60)
                success = run_full_pipeline()
                break
            elif choice == '3':
                print("\n" + "="*60)
                success = run_diffusion_pipeline()
                break
            elif choice == '4':
                print("👋 Goodbye!")
                return 0
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user. Goodbye!")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())