#!/usr/bin/env python3
"""
Test suite for the unified main.py entry point with PyTorch support.

This module tests the enhanced main.py functionality including:
1. Framework selection (TensorFlow/PyTorch)
2. PyTorch-specific commands (diffusion training, sampling, etc.)
3. Model migration utilities
4. Configuration management
5. Command-line interface extensions
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import argparse
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock the problematic imports to allow testing
sys.modules['tensorflow'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.utils'] = Mock()
sys.modules['torch.utils.data'] = Mock()

# Mock all the DeepSculpt modules
mock_modules = [
    'models', 'trainer', 'workflow', 'api', 'bot',
    'pytorch_models', 'pytorch_trainer', 'pytorch_collector',
    'pytorch_curator', 'pytorch_sculptor', 'pytorch_diffusion',
    'pytorch_workflow', 'pytorch_mlflow_tracking', 'pytorch_utils',
    'pytorch_visualization'
]

for module in mock_modules:
    sys.modules[module] = Mock()

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

# Now try to import main with mocked dependencies
try:
    # Patch the availability flags before importing
    with patch.dict('sys.modules', {
        'tensorflow': Mock(),
        'torch': Mock(),
        'models': Mock(),
        'trainer': Mock(),
        'workflow': Mock(),
        'api': Mock(),
        'bot': Mock(),
        'pytorch_models': Mock(),
        'pytorch_trainer': Mock(),
        'pytorch_collector': Mock(),
        'pytorch_curator': Mock(),
        'pytorch_sculptor': Mock(),
        'pytorch_diffusion': Mock(),
        'pytorch_workflow': Mock(),
        'pytorch_mlflow_tracking': Mock(),
        'pytorch_utils': Mock(),
        'pytorch_visualization': Mock()
    }):
        import main
        from main import parse_arguments
        MAIN_AVAILABLE = True
except ImportError as e:
    MAIN_AVAILABLE = False
    pytest.skip(f"Could not import main module: {e}", allow_module_level=True)


class TestArgumentParsing:
    """Test command-line argument parsing for unified interface."""
    
    def test_train_command_tensorflow(self):
        """Test training command with TensorFlow framework."""
        args = parse_arguments([
            'train', '--framework=tensorflow', '--model-type=skip',
            '--epochs=10', '--batch-size=16'
        ])
        
        assert args.command == 'train'
        assert args.framework == 'tensorflow'
        assert args.model_type == 'skip'
        assert args.epochs == 10
        assert args.batch_size == 16
    
    def test_train_command_pytorch(self):
        """Test training command with PyTorch framework."""
        args = parse_arguments([
            'train', '--framework=pytorch', '--model-type=complex',
            '--epochs=50', '--sparse', '--mixed-precision'
        ])
        
        assert args.command == 'train'
        assert args.framework == 'pytorch'
        assert args.model_type == 'complex'
        assert args.epochs == 50
        assert args.sparse is True
        assert args.mixed_precision is True
    
    def test_diffusion_training_command(self):
        """Test diffusion training command."""
        args = parse_arguments([
            'train-diffusion', '--epochs=100', '--timesteps=1000',
            '--noise-schedule=cosine', '--sparse'
        ])
        
        assert args.command == 'train-diffusion'
        assert args.epochs == 100
        assert args.timesteps == 1000
        assert args.noise_schedule == 'cosine'
        assert args.sparse is True
    
    def test_diffusion_sampling_command(self):
        """Test diffusion sampling command."""
        args = parse_arguments([
            'sample-diffusion', '--checkpoint=/path/to/model.pt',
            '--num-samples=20', '--visualize'
        ])
        
        assert args.command == 'sample-diffusion'
        assert args.checkpoint == '/path/to/model.pt'
        assert args.num_samples == 20
        assert args.visualize is True
    
    def test_model_migration_command(self):
        """Test model migration command."""
        args = parse_arguments([
            'migrate-model', '--tf-checkpoint=/path/to/tf_model',
            '--pytorch-output=/path/to/pytorch_model', '--validate'
        ])
        
        assert args.command == 'migrate-model'
        assert args.tf_checkpoint == '/path/to/tf_model'
        assert args.pytorch_output == '/path/to/pytorch_model'
        assert args.validate is True
    
    def test_data_generation_command(self):
        """Test data generation command."""
        args = parse_arguments([
            'generate-data', '--num-samples=500', '--void-dim=32',
            '--sparse', '--sparse-threshold=0.2'
        ])
        
        assert args.command == 'generate-data'
        assert args.num_samples == 500
        assert args.void_dim == 32
        assert args.sparse is True
        assert args.sparse_threshold == 0.2
    
    def test_model_evaluation_command(self):
        """Test model evaluation command."""
        args = parse_arguments([
            'evaluate', '--checkpoint=/path/to/model.pt',
            '--model-type=diffusion', '--num-samples=5', '--visualize'
        ])
        
        assert args.command == 'evaluate'
        assert args.checkpoint == '/path/to/model.pt'
        assert args.model_type == 'diffusion'
        assert args.num_samples == 5
        assert args.visualize is True
    
    def test_workflow_command_with_framework(self):
        """Test workflow command with framework selection."""
        args = parse_arguments([
            'workflow', '--framework=pytorch', '--mode=production',
            '--model-type=skip'
        ])
        
        assert args.command == 'workflow'
        assert args.framework == 'pytorch'
        assert args.mode == 'production'
        assert args.model_type == 'skip'


class TestFrameworkAvailability:
    """Test framework availability checking."""
    
    @patch('main.PYTORCH_AVAILABLE', False)
    def test_pytorch_unavailable_error(self):
        """Test error when PyTorch is not available."""
        args = argparse.Namespace(framework='pytorch', command='train')
        
        with patch('main.parse_arguments', return_value=args):
            result = main.main()
            assert result == 1
    
    @patch('main.TF_AVAILABLE', False)
    def test_tensorflow_unavailable_error(self):
        """Test error when TensorFlow is not available."""
        args = argparse.Namespace(framework='tensorflow', command='train')
        
        with patch('main.parse_arguments', return_value=args):
            result = main.main()
            assert result == 1


class TestTrainingFunctions:
    """Test training function routing and execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            model_type='skip',
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.999,
            void_dim=32,
            noise_dim=50,
            color=True,
            snapshot_freq=1,
            data_folder=self.temp_dir,
            output_dir=self.temp_dir,
            mlflow=False,
            verbose=False,
            sparse=False,
            mixed_precision=False,
            num_samples=10,
            cpu=True,
            generate_samples=False
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_model_routing_pytorch(self):
        """Test that train_model routes to PyTorch when framework=pytorch."""
        self.args.framework = 'pytorch'
        
        with patch('main.train_pytorch_model') as mock_pytorch_train:
            mock_pytorch_train.return_value = 0
            result = train_model(self.args)
            
            mock_pytorch_train.assert_called_once_with(self.args)
            assert result == 0
    
    def test_train_model_routing_tensorflow(self):
        """Test that train_model routes to TensorFlow when framework=tensorflow."""
        self.args.framework = 'tensorflow'
        
        with patch('main.train_tensorflow_model') as mock_tf_train:
            mock_tf_train.return_value = 0
            result = train_model(self.args)
            
            mock_tf_train.assert_called_once_with(self.args)
            assert result == 0
    
    @patch('main.PYTORCH_AVAILABLE', True)
    @patch('main.torch')
    @patch('main.PyTorchCollector')
    @patch('main.PyTorchModelFactory')
    @patch('main.GANTrainer')
    def test_pytorch_training_setup(self, mock_trainer, mock_factory, mock_collector, mock_torch):
        """Test PyTorch training setup and configuration."""
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.utils.data.DataLoader.return_value = Mock()
        
        # Mock model factory
        mock_generator = Mock()
        mock_discriminator = Mock()
        mock_factory.create_gan_generator.return_value = mock_generator
        mock_factory.create_gan_discriminator.return_value = mock_discriminator
        
        # Mock collector
        mock_dataset = Mock()
        mock_collector_instance = Mock()
        mock_collector_instance.create_streaming_dataset.return_value = mock_dataset
        mock_collector.return_value = mock_collector_instance
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = {}
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock torch.save
        with patch('main.torch.save'):
            result = train_pytorch_model(self.args)
        
        # Verify calls
        mock_factory.create_gan_generator.assert_called_once()
        mock_factory.create_gan_discriminator.assert_called_once()
        mock_collector.assert_called_once()
        mock_trainer.assert_called_once()
        
        assert result == 0


class TestDiffusionFunctions:
    """Test diffusion model training and sampling functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.train_args = argparse.Namespace(
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            weight_decay=1e-4,
            void_dim=32,
            timesteps=100,
            noise_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            data_folder=self.temp_dir,
            output_dir=self.temp_dir,
            sparse=False,
            mixed_precision=False,
            num_samples=5,
            cpu=True,
            mlflow=False
        )
        
        self.sample_args = argparse.Namespace(
            checkpoint=os.path.join(self.temp_dir, 'model.pt'),
            num_samples=3,
            num_steps=10,
            output_dir=self.temp_dir,
            visualize=False,
            cpu=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('main.PYTORCH_AVAILABLE', True)
    @patch('main.torch')
    @patch('main.PyTorchCollector')
    @patch('main.PyTorchModelFactory')
    @patch('main.NoiseScheduler')
    @patch('main.Diffusion3DPipeline')
    @patch('main.DiffusionTrainer')
    def test_diffusion_training_setup(self, mock_trainer, mock_pipeline, mock_scheduler,
                                    mock_factory, mock_collector, mock_torch):
        """Test diffusion model training setup."""
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.utils.data.DataLoader.return_value = Mock()
        
        # Mock components
        mock_model = Mock()
        mock_factory.create_diffusion_model.return_value = mock_model
        
        mock_dataset = Mock()
        mock_collector_instance = Mock()
        mock_collector_instance.create_streaming_dataset.return_value = mock_dataset
        mock_collector.return_value = mock_collector_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = {}
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock torch.save
        with patch('main.torch.save'):
            result = train_diffusion_model(self.train_args)
        
        # Verify calls
        mock_factory.create_diffusion_model.assert_called_once()
        mock_scheduler.assert_called_once()
        mock_pipeline.assert_called_once()
        mock_trainer.assert_called_once()
        
        assert result == 0
    
    @patch('main.PYTORCH_AVAILABLE', True)
    @patch('main.torch')
    @patch('main.PyTorchModelFactory')
    @patch('main.Diffusion3DPipeline')
    def test_diffusion_sampling(self, mock_pipeline, mock_factory, mock_torch):
        """Test diffusion model sampling."""
        # Create mock checkpoint
        checkpoint_data = {
            'model_state_dict': {},
            'noise_scheduler': Mock(),
            'config': {
                'void_dim': 32,
                'timesteps': 100,
                'sparse': False
            }
        }
        
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.load.return_value = checkpoint_data
        mock_torch.save = Mock()
        
        # Mock model
        mock_model = Mock()
        mock_model.load_state_dict = Mock()
        mock_model.eval = Mock()
        mock_factory.create_diffusion_model.return_value = mock_model
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_sample = Mock()
        mock_sample.cpu.return_value = Mock()
        mock_pipeline_instance.sample.return_value = mock_sample
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create checkpoint file
        with open(self.sample_args.checkpoint, 'w') as f:
            json.dump({}, f)
        
        result = sample_diffusion_model(self.sample_args)
        
        # Verify calls
        mock_torch.load.assert_called_once()
        mock_factory.create_diffusion_model.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        mock_pipeline.assert_called_once()
        
        assert result == 0


class TestUtilityFunctions:
    """Test utility functions for data generation, evaluation, and comparison."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('main.PYTORCH_AVAILABLE', True)
    @patch('main.torch')
    @patch('main.PyTorchCollector')
    def test_data_generation(self, mock_collector, mock_torch):
        """Test PyTorch data generation."""
        args = argparse.Namespace(
            num_samples=10,
            void_dim=32,
            num_shapes=3,
            output_dir=self.temp_dir,
            sparse=False,
            sparse_threshold=0.1,
            cpu=True
        )
        
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        
        # Mock collector
        mock_collector_instance = Mock()
        mock_collector_instance.create_collection.return_value = ['path1', 'path2']
        mock_collector.return_value = mock_collector_instance
        
        result = generate_pytorch_data(args)
        
        # Verify calls
        mock_collector.assert_called_once()
        mock_collector_instance.create_collection.assert_called_once_with(10)
        
        # Check metadata file was created
        metadata_path = os.path.join(self.temp_dir, 'dataset_metadata.json')
        assert os.path.exists(metadata_path)
        
        assert result == 0
    
    @patch('main.PYTORCH_AVAILABLE', True)
    @patch('main.torch')
    @patch('main.PyTorchModelFactory')
    def test_model_evaluation(self, mock_factory, mock_torch):
        """Test model evaluation functionality."""
        args = argparse.Namespace(
            checkpoint=os.path.join(self.temp_dir, 'model.pt'),
            model_type='gan',
            num_samples=3,
            output_dir=self.temp_dir,
            visualize=False,
            cpu=True
        )
        
        # Create mock config file
        config = {
            'model_type': 'skip',
            'void_dim': 32,
            'noise_dim': 50,
            'color_mode': 1,
            'sparse': False
        }
        config_path = os.path.join(self.temp_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.load.return_value = {}
        mock_torch.save = Mock()
        mock_torch.randn.return_value = Mock()
        mock_torch.cat.return_value = Mock()
        
        # Mock model
        mock_model = Mock()
        mock_model.load_state_dict = Mock()
        mock_model.eval = Mock()
        mock_sample = Mock()
        mock_sample.cpu.return_value = Mock()
        mock_model.return_value = mock_sample
        mock_factory.create_gan_generator.return_value = mock_model
        
        result = evaluate_pytorch_model(args)
        
        # Verify calls
        mock_factory.create_gan_generator.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        
        # Check results file was created
        results_path = os.path.join(self.temp_dir, 'evaluation_results.json')
        assert os.path.exists(results_path)
        
        assert result == 0


class TestWorkflowIntegration:
    """Test workflow integration with framework selection."""
    
    def test_workflow_routing_pytorch(self):
        """Test workflow routing to PyTorch implementation."""
        args = argparse.Namespace(
            framework='pytorch',
            mode='development',
            data_folder='./data',
            model_type='skip',
            epochs=10,
            schedule=False
        )
        
        with patch('main.PYTORCH_AVAILABLE', True):
            with patch('main.run_pytorch_workflow') as mock_pytorch_workflow:
                mock_pytorch_workflow.return_value = 0
                result = run_workflow(args)
                
                mock_pytorch_workflow.assert_called_once_with(args)
                assert result == 0
    
    def test_workflow_routing_tensorflow(self):
        """Test workflow routing to TensorFlow implementation."""
        args = argparse.Namespace(
            framework='tensorflow',
            mode='development'
        )
        
        with patch('main.run_tensorflow_workflow') as mock_tf_workflow:
            mock_tf_workflow.return_value = 0
            result = run_workflow(args)
            
            mock_tf_workflow.assert_called_once_with(args)
            assert result == 0
    
    @patch('main.PYTORCH_AVAILABLE', True)
    @patch('main.PyTorchManager')
    def test_pytorch_workflow_execution(self, mock_manager):
        """Test PyTorch workflow execution."""
        args = argparse.Namespace(
            mode='development',
            data_folder='./data',
            model_type='skip',
            epochs=10
        )
        
        # Mock manager
        mock_manager_instance = Mock()
        mock_manager_instance.run_full_experiment.return_value = {'status': 'success'}
        mock_manager.return_value = mock_manager_instance
        
        result = run_pytorch_workflow(args)
        
        # Verify calls
        mock_manager.assert_called_once()
        mock_manager_instance.run_full_experiment.assert_called_once()
        
        assert result == 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        with patch('main.parse_arguments') as mock_parse:
            mock_parse.return_value = argparse.Namespace(command='invalid')
            result = main.main()
            assert result == 1
    
    @patch('main.PYTORCH_AVAILABLE', False)
    def test_pytorch_unavailable_for_diffusion(self):
        """Test error when PyTorch is unavailable for diffusion training."""
        args = argparse.Namespace(command='train-diffusion')
        
        with patch('main.parse_arguments', return_value=args):
            result = main.main()
            assert result == 1
    
    @patch('main.TF_AVAILABLE', False)
    @patch('main.PYTORCH_AVAILABLE', False)
    def test_both_frameworks_unavailable_for_migration(self):
        """Test error when both frameworks are unavailable for migration."""
        args = argparse.Namespace(
            command='migrate-model',
            tf_checkpoint='./tf_model',
            pytorch_output='./pytorch_model'
        )
        
        result = migrate_tensorflow_model(args)
        assert result == 1


if __name__ == '__main__':
    pytest.main([__file__])