#!/usr/bin/env python3
"""
Unit Tests for DeepSculpt v2.0 Pipeline

Tests for the end-to-end pipeline functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import json

# Import pipeline components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline import DeepSculptPipeline


class TestDeepSculptPipeline:
    """Test cases for the DeepSculpt pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_config(self, temp_dir):
        """Create minimal configuration for testing."""
        return {
            'model_type': 'gan',
            'gan_model_type': 'simple',
            'void_dim': 16,  # Very small for testing
            'noise_dim': 32,
            'epochs': 1,
            'batch_size': 2,
            'num_samples': 4,  # Very small dataset
            'num_shapes': 2,
            'sparse_mode': False,
            'output_dir': str(temp_dir),
            'log_level': 'WARNING',  # Reduce log noise
            'enable_monitoring': False,
            'enable_optimization': False,
            'num_eval_samples': 2
        }
    
    def test_pipeline_initialization(self, minimal_config):
        """Test pipeline initialization."""
        pipeline = DeepSculptPipeline(minimal_config)
        
        assert pipeline.config == minimal_config
        assert pipeline.device in ['cuda', 'cpu']
        assert pipeline.base_dir.exists()
        assert pipeline.data_dir.exists()
        assert pipeline.models_dir.exists()
    
    def test_stage_1_data_generation(self, minimal_config):
        """Test data generation stage."""
        pipeline = DeepSculptPipeline(minimal_config)
        
        success = pipeline.stage_1_generate_data()
        assert success
        assert pipeline.pipeline_state['data_generated']
        
        # Check that data files were created
        data_files = list(pipeline.data_dir.glob("*.pt"))
        assert len(data_files) == minimal_config['num_samples']
        
        # Check metadata file
        metadata_file = pipeline.data_dir / "generation_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        assert metadata['num_samples'] == minimal_config['num_samples']
    
    def test_stage_2_data_preprocessing(self, minimal_config):
        """Test data preprocessing stage."""
        pipeline = DeepSculptPipeline(minimal_config)
        
        # First generate data
        pipeline.stage_1_generate_data()
        
        # Then preprocess
        success = pipeline.stage_2_preprocess_data()
        assert success
        assert pipeline.pipeline_state['data_preprocessed']
        
        # Check split file
        split_file = pipeline.data_dir / "data_split.json"
        assert split_file.exists()
        
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        assert 'train_paths' in split_info
        assert 'val_paths' in split_info
    
    @pytest.mark.slow
    def test_stage_3_gan_training(self, minimal_config):
        """Test GAN training stage."""
        pipeline = DeepSculptPipeline(minimal_config)
        
        # Setup data
        pipeline.stage_1_generate_data()
        pipeline.stage_2_preprocess_data()
        
        # Train model
        success = pipeline.stage_3_train_model()
        assert success
        assert pipeline.pipeline_state['model_trained']
        
        # Check model files
        generator_file = pipeline.models_dir / "generator_final.pt"
        discriminator_file = pipeline.models_dir / "discriminator_final.pt"
        assert generator_file.exists()
        assert discriminator_file.exists()
    
    def test_config_validation(self, minimal_config):
        """Test configuration validation."""
        # Test with invalid config
        invalid_config = minimal_config.copy()
        invalid_config['void_dim'] = -1
        
        # Pipeline should still initialize but may fail during execution
        pipeline = DeepSculptPipeline(invalid_config)
        assert pipeline.config['void_dim'] == -1
    
    def test_pipeline_state_tracking(self, minimal_config):
        """Test pipeline state tracking."""
        pipeline = DeepSculptPipeline(minimal_config)
        
        # Initial state
        assert not pipeline.pipeline_state['data_generated']
        assert not pipeline.pipeline_state['data_preprocessed']
        
        # After data generation
        pipeline.stage_1_generate_data()
        assert pipeline.pipeline_state['data_generated']
        assert not pipeline.pipeline_state['data_preprocessed']
        
        # After preprocessing
        pipeline.stage_2_preprocess_data()
        assert pipeline.pipeline_state['data_generated']
        assert pipeline.pipeline_state['data_preprocessed']
    
    def test_directory_structure(self, minimal_config):
        """Test that pipeline creates proper directory structure."""
        pipeline = DeepSculptPipeline(minimal_config)
        
        expected_dirs = [
            pipeline.data_dir,
            pipeline.models_dir,
            pipeline.samples_dir,
            pipeline.results_dir,
            pipeline.logs_dir,
            pipeline.visualizations_dir
        ]
        
        for directory in expected_dirs:
            assert directory.exists()
            assert directory.is_dir()
    
    def test_config_saving(self, minimal_config):
        """Test configuration saving."""
        pipeline = DeepSculptPipeline(minimal_config)
        pipeline.save_config()
        
        config_file = pipeline.base_dir / "pipeline_config.yaml"
        assert config_file.exists()


class TestPipelineUtilities:
    """Test utility functions for pipeline."""
    
    def test_create_config_from_args(self):
        """Test configuration creation from arguments."""
        from pipeline import create_config_from_args
        
        # Mock arguments object
        class MockArgs:
            model_type = 'gan'
            gan_model_type = 'skip'
            void_dim = 64
            noise_dim = 100
            epochs = 50
            batch_size = 32
            learning_rate = 0.0002
            mixed_precision = True
            num_samples = 1000
            num_shapes = 5
            sparse = True
            sparse_threshold = 0.1
            output_dir = './test_output'
            log_level = 'INFO'
            enable_monitoring = True
            enable_optimization = True
            num_eval_samples = 10
            viz_backend = 'plotly'
        
        config = create_config_from_args(MockArgs())
        
        assert config['model_type'] == 'gan'
        assert config['void_dim'] == 64
        assert config['epochs'] == 50
        assert config['sparse_mode'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])