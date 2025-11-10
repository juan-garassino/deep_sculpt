"""
Tests for PyTorch tensor support in visualization.py

This module tests the enhanced visualization functionality that supports
both numpy arrays and PyTorch tensors, ensuring output equivalence
and proper device handling.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import the visualization module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from visualization import Visualizer, PyTorchVisualizer, _tensor_to_numpy, _is_torch_tensor, _get_tensor_device

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class TestTensorUtilities:
    """Test utility functions for tensor handling."""
    
    def test_tensor_to_numpy_with_numpy(self):
        """Test tensor_to_numpy with numpy array input."""
        arr = np.random.rand(3, 3, 3)
        result = _tensor_to_numpy(arr)
        np.testing.assert_array_equal(result, arr)
        assert isinstance(result, np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_tensor_to_numpy_with_cpu_tensor(self):
        """Test tensor_to_numpy with CPU PyTorch tensor."""
        arr = np.random.rand(3, 3, 3)
        tensor = torch.from_numpy(arr).float()
        result = _tensor_to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, arr, decimal=6)
        assert isinstance(result, np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                       reason="PyTorch or CUDA not available")
    def test_tensor_to_numpy_with_gpu_tensor(self):
        """Test tensor_to_numpy with GPU PyTorch tensor."""
        arr = np.random.rand(3, 3, 3)
        tensor = torch.from_numpy(arr).float().cuda()
        result = _tensor_to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, arr, decimal=6)
        assert isinstance(result, np.ndarray)
    
    def test_is_torch_tensor_with_numpy(self):
        """Test _is_torch_tensor with numpy array."""
        arr = np.random.rand(3, 3, 3)
        assert not _is_torch_tensor(arr)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_is_torch_tensor_with_tensor(self):
        """Test _is_torch_tensor with PyTorch tensor."""
        tensor = torch.rand(3, 3, 3)
        assert _is_torch_tensor(tensor)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_tensor_device_cpu(self):
        """Test _get_tensor_device with CPU tensor."""
        tensor = torch.rand(3, 3, 3)
        device = _get_tensor_device(tensor)
        assert "cpu" in device
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                       reason="PyTorch or CUDA not available")
    def test_get_tensor_device_gpu(self):
        """Test _get_tensor_device with GPU tensor."""
        tensor = torch.rand(3, 3, 3).cuda()
        device = _get_tensor_device(tensor)
        assert "cuda" in device


class TestVisualizerPyTorchSupport:
    """Test PyTorch tensor support in base Visualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer(figsize=10)
        
        # Create test data
        self.structure = np.zeros((8, 8, 8))
        self.structure[2:6, 2:6, 2:6] = 1  # Create a cube
        
        self.colors = np.zeros(self.structure.shape + (4,))  # RGBA
        self.colors[self.structure == 1] = [1, 0, 0, 1]  # Red color
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_sections_with_numpy(self, mock_close, mock_show):
        """Test plot_sections with numpy array."""
        fig = self.visualizer.plot_sections(self.structure, show=False)
        assert fig is not None
        mock_close.assert_called_once()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_sections_with_tensor(self, mock_close, mock_show):
        """Test plot_sections with PyTorch tensor."""
        tensor = torch.from_numpy(self.structure).float()
        fig = self.visualizer.plot_sections(tensor, show=False)
        assert fig is not None
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_sculpture_with_numpy(self, mock_close, mock_show):
        """Test plot_sculpture with numpy arrays."""
        fig = self.visualizer.plot_sculpture(
            self.structure, 
            colors=self.colors,
            show=False,
            angles=[0, 1]
        )
        assert fig is not None
        mock_close.assert_called_once()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_sculpture_with_tensors(self, mock_close, mock_show):
        """Test plot_sculpture with PyTorch tensors."""
        structure_tensor = torch.from_numpy(self.structure).float()
        colors_tensor = torch.from_numpy(self.colors).float()
        
        fig = self.visualizer.plot_sculpture(
            structure_tensor,
            colors=colors_tensor,
            show=False,
            angles=[0, 1]
        )
        assert fig is not None
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_single_view_with_numpy(self, mock_close, mock_show):
        """Test plot_single_view with numpy arrays."""
        fig = self.visualizer.plot_single_view(
            self.structure,
            colors=self.colors,
            show=False
        )
        assert fig is not None
        mock_close.assert_called_once()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_single_view_with_tensors(self, mock_close, mock_show):
        """Test plot_single_view with PyTorch tensors."""
        structure_tensor = torch.from_numpy(self.structure).float()
        colors_tensor = torch.from_numpy(self.colors).float()
        
        fig = self.visualizer.plot_single_view(
            structure_tensor,
            colors=colors_tensor,
            show=False
        )
        assert fig is not None
        mock_close.assert_called_once()
    
    def test_voxel_to_pointcloud_with_numpy(self):
        """Test voxel_to_pointcloud with numpy array."""
        points = self.visualizer.voxel_to_pointcloud(self.structure, subdivision=2)
        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 3  # Should have x, y, z coordinates
        assert len(points) > 0  # Should have some points
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_voxel_to_pointcloud_with_tensor(self):
        """Test voxel_to_pointcloud with PyTorch tensor."""
        tensor = torch.from_numpy(self.structure).float()
        points = self.visualizer.voxel_to_pointcloud(tensor, subdivision=2)
        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 3  # Should have x, y, z coordinates
        assert len(points) > 0  # Should have some points
    
    @patch('plotly.graph_objects.Figure.show')
    def test_plot_pointcloud_with_numpy(self, mock_show):
        """Test plot_pointcloud with numpy arrays."""
        points = self.visualizer.voxel_to_pointcloud(self.structure, subdivision=2)
        colors = np.array([[255, 0, 0]] * len(points))  # Red points
        
        fig = self.visualizer.plot_pointcloud(points, colors=colors, show=False)
        assert fig is not None
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('plotly.graph_objects.Figure.show')
    def test_plot_pointcloud_with_tensors(self, mock_show):
        """Test plot_pointcloud with PyTorch tensors."""
        structure_tensor = torch.from_numpy(self.structure).float()
        points = self.visualizer.voxel_to_pointcloud(structure_tensor, subdivision=2)
        colors_tensor = torch.tensor([[255, 0, 0]] * len(points)).float()
        
        fig = self.visualizer.plot_pointcloud(points, colors=colors_tensor, show=False)
        assert fig is not None


class TestPyTorchVisualizer:
    """Test PyTorchVisualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = PyTorchVisualizer(figsize=10)
        
        # Create test data
        self.structure = np.zeros((8, 8, 8))
        self.structure[2:6, 2:6, 2:6] = 1  # Create a cube
    
    def test_initialization(self):
        """Test PyTorchVisualizer initialization."""
        assert isinstance(self.visualizer, PyTorchVisualizer)
        assert isinstance(self.visualizer, Visualizer)  # Should inherit from base class
        assert self.visualizer.backend == "matplotlib"
        assert self.visualizer.device in ["cpu", "cuda"]
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_training_progress(self, mock_close, mock_show):
        """Test plot_training_progress method."""
        metrics = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'accuracy': [0.2, 0.4, 0.6, 0.8, 0.9]
        }
        
        fig = self.visualizer.plot_training_progress(metrics, show=False)
        assert fig is not None
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_visualize_latent_space_placeholder(self, mock_close, mock_show):
        """Test visualize_latent_space placeholder implementation."""
        if TORCH_AVAILABLE:
            # Create a dummy model
            model = torch.nn.Linear(10, 5)
            fig = self.visualizer.visualize_latent_space(model, show=False)
            assert fig is not None
            mock_close.assert_called_once()


class TestFileHandling:
    """Test file loading and saving with PyTorch tensors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer(figsize=10)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.structure = np.zeros((8, 8, 8))
        self.structure[2:6, 2:6, 2:6] = 1
        self.colors = np.zeros(self.structure.shape + (4,))
        self.colors[self.structure == 1] = [1, 0, 0, 1]
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_visualize_sample_from_numpy_files(self, mock_close, mock_show):
        """Test loading and visualizing numpy files."""
        # Save test data
        structure_path = os.path.join(self.temp_dir, "structure_001.npy")
        colors_path = os.path.join(self.temp_dir, "colors_001.npy")
        
        np.save(structure_path, self.structure)
        np.save(colors_path, self.colors)
        
        # Test visualization
        fig = self.visualizer.visualize_sample_from_files(
            structure_path, colors_path, show=False
        )
        assert fig is not None
        mock_close.assert_called_once()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_visualize_sample_from_pytorch_files(self, mock_close, mock_show):
        """Test loading and visualizing PyTorch tensor files."""
        # Save test data as tensors
        structure_path = os.path.join(self.temp_dir, "structure_001.pt")
        colors_path = os.path.join(self.temp_dir, "colors_001.pt")
        
        structure_tensor = torch.from_numpy(self.structure).float()
        colors_tensor = torch.from_numpy(self.colors).float()
        
        torch.save(structure_tensor, structure_path)
        torch.save(colors_tensor, colors_path)
        
        # Test visualization
        fig = self.visualizer.visualize_sample_from_files(
            structure_path, colors_path, show=False
        )
        assert fig is not None
        mock_close.assert_called_once()


class TestOutputEquivalence:
    """Test that PyTorch tensor and numpy array inputs produce equivalent outputs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer(figsize=10)
        
        # Create test data
        np.random.seed(42)  # For reproducible tests
        self.structure = np.random.randint(0, 2, (8, 8, 8))
        self.colors = np.random.rand(*self.structure.shape, 4)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pointcloud_conversion_equivalence(self):
        """Test that pointcloud conversion produces equivalent results."""
        # Convert with numpy
        points_numpy = self.visualizer.voxel_to_pointcloud(self.structure, subdivision=2)
        
        # Convert with tensor
        tensor = torch.from_numpy(self.structure).float()
        points_tensor = self.visualizer.voxel_to_pointcloud(tensor, subdivision=2)
        
        # Results should be identical
        np.testing.assert_array_equal(points_numpy, points_tensor)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_sculpture_plot_equivalence(self, mock_close, mock_savefig):
        """Test that sculpture plots produce equivalent results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Plot with numpy
            numpy_path = os.path.join(temp_dir, "numpy_plot.png")
            fig_numpy = self.visualizer.plot_sculpture(
                self.structure, 
                colors=self.colors,
                save_path=numpy_path,
                show=False,
                angles=[0]
            )
            
            # Plot with tensor
            tensor_path = os.path.join(temp_dir, "tensor_plot.png")
            structure_tensor = torch.from_numpy(self.structure).float()
            colors_tensor = torch.from_numpy(self.colors).float()
            
            fig_tensor = self.visualizer.plot_sculpture(
                structure_tensor,
                colors=colors_tensor,
                save_path=tensor_path,
                show=False,
                angles=[0]
            )
            
            # Both should produce valid figures
            assert fig_numpy is not None
            assert fig_tensor is not None
            
            # savefig should be called twice
            assert mock_savefig.call_count == 2


if __name__ == "__main__":
    # Run basic tests
    print("Running basic visualization tests...")
    
    # Test utility functions
    test_utils = TestTensorUtilities()
    test_utils.test_tensor_to_numpy_with_numpy()
    test_utils.test_is_torch_tensor_with_numpy()
    
    if TORCH_AVAILABLE:
        test_utils.test_tensor_to_numpy_with_cpu_tensor()
        test_utils.test_is_torch_tensor_with_tensor()
        test_utils.test_get_tensor_device_cpu()
        print("PyTorch tensor utility tests passed!")
    
    # Test visualizer
    test_viz = TestVisualizerPyTorchSupport()
    test_viz.setup_method()
    
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
        test_viz.test_plot_sections_with_numpy()
        test_viz.test_plot_sculpture_with_numpy()
        test_viz.test_plot_single_view_with_numpy()
        
        if TORCH_AVAILABLE:
            test_viz.test_plot_sections_with_tensor()
            test_viz.test_plot_sculpture_with_tensors()
            test_viz.test_plot_single_view_with_tensors()
            print("PyTorch tensor visualization tests passed!")
    
    test_viz.test_voxel_to_pointcloud_with_numpy()
    if TORCH_AVAILABLE:
        test_viz.test_voxel_to_pointcloud_with_tensor()
    
    print("All basic tests passed!")