"""
Tests for PyTorch-enhanced workflow functionality.

This module tests the enhanced workflow capabilities including:
- PyTorch model integration
- Enhanced MLflow tracking
- Model comparison utilities
- Framework switching
- Diffusion model support
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the deepSculpt directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import modules to test
from workflow import Manager, PyTorchManager, build_flow, build_pytorch_flow
from workflow import preprocess_data, evaluate_model, train_model, compare_and_promote

if TORCH_AVAILABLE:
    from pytorch_mlflow_tracking import PyTorchMLflowTracker, create_pytorch_mlflow_tracker


class TestPyTorchManager:
    """Test the enhanced PyTorch Manager class."""
    
    def test_manager_initialization_pytorch(self):
        """Test PyTorch manager initialization."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        manager = PyTorchManager(framework="pytorch")
        assert manager.framework == "pytorch"
        assert manager.device is not None
        assert "pytorch" in manager.comment
    
    def test_manager_initialization_tensorflow(self):
        """Test TensorFlow manager initialization."""
        manager = PyTorchManager(framework="tensorflow")
        assert manager.framework == "tensorflow"
        assert manager.device is None
        assert "tensorflow" in manager.comment
    
    def test_backward_compatibility(self):
        """Test that Manager alias works for backward compatibility."""
        manager = Manager()
        assert hasattr(manager, 'framework')
        assert manager.framework == "tensorflow"  # Default
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_data_loading(self):
        """Test PyTorch tensor data loading."""
        import numpy as np
        
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            volume_path = os.path.join(temp_dir, "volume_data.npy")
            material_path = os.path.join(temp_dir, "material_data.npy")
            
            # Create dummy data
            volume_data = np.random.rand(10, 32, 32, 32)
            material_data = np.random.rand(10, 32, 32, 32, 3)
            
            np.save(volume_path, volume_data)
            np.save(material_path, material_data)
            
            # Test loading
            manager = PyTorchManager(framework="pytorch")
            volumes, materials = manager.load_pytorch_data(volume_path, material_path)
            
            assert torch.is_tensor(volumes)
            assert torch.is_tensor(materials)
            assert volumes.shape == (10, 32, 32, 32)
            assert materials.shape == (10, 32, 32, 32, 3)
    
    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not available")
    def test_enhanced_mlflow_model_saving(self):
        """Test enhanced MLflow model saving with PyTorch."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Create a simple PyTorch model
        model = nn.Linear(10, 1)
        
        # Mock MLflow operations
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.pytorch.log_model') as mock_log_model, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics:
            
            mock_start_run.return_value.__enter__ = Mock()
            mock_start_run.return_value.__exit__ = Mock()
            
            manager = PyTorchManager(framework="pytorch")
            manager.save_pytorch_model(
                model=model,
                metrics={"loss": 0.5},
                params={"lr": 0.001},
                model_type="test_model"
            )
            
            # Verify MLflow calls
            mock_log_model.assert_called_once()
            mock_log_params.assert_called_once()
            mock_log_metrics.assert_called_once()


class TestPyTorchMLflowTracker:
    """Test the enhanced MLflow tracking for PyTorch models."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not MLFLOW_AVAILABLE, reason="PyTorch or MLflow not available")
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = PyTorchMLflowTracker(experiment_name="test_experiment")
        assert tracker.experiment_name == "test_experiment"
        assert tracker.default_tags["framework"] == "pytorch"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not MLFLOW_AVAILABLE, reason="PyTorch or MLflow not available")
    def test_model_architecture_logging(self):
        """Test model architecture logging."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params:
            
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value = mock_run
            
            tracker = PyTorchMLflowTracker()
            tracker.start_run()
            tracker.log_model_architecture(model, "test_model")
            
            # Verify that parameters were logged
            mock_log_params.assert_called()
            
            # Check that the logged parameters include model info
            call_args = mock_log_params.call_args[0][0]
            assert "test_model_total_parameters" in call_args
            assert "test_model_architecture" in call_args
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not MLFLOW_AVAILABLE, reason="PyTorch or MLflow not available")
    def test_training_metrics_logging(self):
        """Test training metrics logging with GPU memory tracking."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value = mock_run
            
            tracker = PyTorchMLflowTracker()
            tracker.start_run()
            
            metrics = {"loss": 0.5, "accuracy": 0.8}
            tracker.log_training_metrics(metrics, step=1)
            
            # Verify metrics were logged
            mock_log_metrics.assert_called()
            
            # Check that GPU memory metrics were added
            call_args = mock_log_metrics.call_args[0][0]
            assert "loss" in call_args
            assert "accuracy" in call_args
            assert "gpu_memory_allocated_gb" in call_args
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not MLFLOW_AVAILABLE, reason="PyTorch or MLflow not available")
    def test_generation_samples_logging(self):
        """Test logging of generated samples."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_artifact') as mock_log_artifact:
            
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value = mock_run
            
            tracker = PyTorchMLflowTracker()
            tracker.start_run()
            
            # Create dummy samples
            samples = torch.randn(4, 32, 32, 32, 6)
            tracker.log_generation_samples(samples, step=1, sample_type="generated")
            
            # Verify artifacts were logged
            mock_log_artifact.assert_called()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not MLFLOW_AVAILABLE, reason="PyTorch or MLflow not available")
    def test_model_comparison_logging(self):
        """Test model comparison logging."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.log_artifact') as mock_log_artifact:
            
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value = mock_run
            
            tracker = PyTorchMLflowTracker()
            tracker.start_run()
            
            pytorch_metrics = {"loss": 0.3, "accuracy": 0.9}
            tensorflow_metrics = {"loss": 0.5, "accuracy": 0.8}
            
            tracker.log_model_comparison(pytorch_metrics, tensorflow_metrics)
            
            # Verify comparison artifacts and metrics were logged
            mock_log_artifact.assert_called()
            mock_log_metrics.assert_called()


class TestWorkflowIntegration:
    """Test the integration of PyTorch components in the workflow."""
    
    def test_build_pytorch_flow(self):
        """Test building PyTorch workflow."""
        flow = build_pytorch_flow(framework="pytorch", training_mode="gan")
        assert flow is not None
        assert "pytorch" in flow.name
        assert "gan" in flow.name
    
    def test_build_diffusion_flow(self):
        """Test building diffusion workflow."""
        flow = build_pytorch_flow(framework="pytorch", training_mode="diffusion")
        assert flow is not None
        assert "diffusion" in flow.name
    
    def test_backward_compatible_flow(self):
        """Test that original build_flow still works."""
        flow = build_flow()
        assert flow is not None
    
    @patch('workflow.PyTorchCollector')
    @patch('workflow.glob.glob')
    def test_preprocess_pytorch_data(self, mock_glob, mock_collector):
        """Test PyTorch data preprocessing."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Mock existing files
        mock_glob.return_value = ["data/volume_data.npy", "data/material_data.npy"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the data creation
            with patch('workflow.Manager.create_data_dataframe') as mock_create_df:
                import pandas as pd
                mock_df = pd.DataFrame({
                    'chunk_idx': [0, 1],
                    'volume_path': ['vol1.npy', 'vol2.npy'],
                    'material_path': ['mat1.npy', 'mat2.npy']
                })
                mock_create_df.return_value = mock_df
                
                result = preprocess_data("test_exp", temp_dir, framework="pytorch")
                
                assert result is not None
                assert "pytorch" in result
    
    def test_enhanced_model_comparison(self):
        """Test enhanced model comparison with additional factors."""
        eval_metrics = {
            "gen_loss": 1.0,
            "framework": "tensorflow",
            "gpu_memory_used": 3.5,
            "sparsity": 0.4
        }
        
        train_metrics = {
            "gen_loss": 0.8,
            "framework": "pytorch",
            "training_mode": "gan",
            "training_time": 300,
            "epochs": 10,
            "model_parameters_gen": 1000000,
            "model_parameters_disc": 500000
        }
        
        # Test with enhanced comparison
        result = compare_and_promote(eval_metrics, train_metrics, threshold=0.1)
        
        # Should promote due to improvement and additional benefits
        assert result is True
    
    def test_diffusion_model_comparison(self):
        """Test comparison for diffusion models."""
        eval_metrics = {
            "diffusion_loss": 0.5,
            "framework": "pytorch"
        }
        
        train_metrics = {
            "diffusion_loss": 0.3,
            "framework": "pytorch",
            "training_mode": "diffusion",
            "model_parameters": 2000000
        }
        
        result = compare_and_promote(eval_metrics, train_metrics, threshold=0.1)
        
        # Should promote due to significant improvement
        assert result is True


class TestErrorHandling:
    """Test error handling in PyTorch workflow components."""
    
    def test_pytorch_unavailable_fallback(self):
        """Test fallback when PyTorch is not available."""
        with patch('workflow.PYTORCH_AVAILABLE', False):
            manager = PyTorchManager(framework="pytorch")
            # Should fall back to TensorFlow behavior
            assert manager.device is None
    
    def test_mlflow_unavailable_handling(self):
        """Test handling when MLflow is not available."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        with patch('workflow.MLFLOW_AVAILABLE', False):
            # Should not raise an error, just skip MLflow operations
            manager = PyTorchManager(framework="pytorch")
            # This should work without MLflow
            assert manager.framework == "pytorch"
    
    def test_model_loading_error_handling(self):
        """Test error handling in model loading."""
        manager = PyTorchManager()
        
        with patch('mlflow.pytorch.load_model', side_effect=Exception("Model not found")):
            model = manager.load_pytorch_model(stage="Production")
            assert model is None


if __name__ == "__main__":
    pytest.main([__file__])