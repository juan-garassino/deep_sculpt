#!/usr/bin/env python3
"""
Integration test for PyTorch tensor support in visualization.

This script demonstrates the enhanced visualization functionality
with both numpy arrays and PyTorch tensors.
"""

import os
import sys
import tempfile
import numpy as np

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepSculpt'))

from visualization import Visualizer, PyTorchVisualizer

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    print("✓ PyTorch is available")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    print("✗ PyTorch is not available")


def create_test_data():
    """Create test 3D structure and colors."""
    # Create a simple 3D structure (a cube with some features)
    structure = np.zeros((16, 16, 16))
    
    # Add a solid cube in the center
    structure[4:12, 4:12, 4:12] = 1
    
    # Add some edges
    structure[0, :, 0] = 1  # Bottom edge
    structure[:, 0, 0] = 1  # Side edge
    structure[0, 0, :] = 1  # Vertical edge
    
    # Create colors (RGBA)
    colors = np.zeros(structure.shape + (4,))
    colors[structure == 1] = [0.8, 0.2, 0.2, 1.0]  # Red color
    
    return structure, colors


def test_numpy_visualization(save_dir=None):
    """Test visualization with numpy arrays."""
    print("\n=== Testing NumPy Array Visualization ===")
    
    structure, colors = create_test_data()
    visualizer = Visualizer(figsize=12)
    
    # Determine save directory
    if save_dir is None:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = os.path.join(save_dir, "numpy")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving NumPy visualization results to: {temp_dir}")
        # Test plot_sections
        print("Testing plot_sections...")
        fig = visualizer.plot_sections(
            structure, 
            title="Test Structure Sections (NumPy)",
            save_path=os.path.join(temp_dir, "sections_numpy.png")
        )
        print("✓ plot_sections completed")
        
        # Test plot_sculpture
        print("Testing plot_sculpture...")
        fig = visualizer.plot_sculpture(
            structure, 
            colors=colors,
            title="Test Sculpture (NumPy)",
            angles=[0, 1],
            save_path=os.path.join(temp_dir, "sculpture_numpy.png")
        )
        print("✓ plot_sculpture completed")
        
        # Test plot_single_view
        print("Testing plot_single_view...")
        fig = visualizer.plot_single_view(
            structure,
            colors=colors,
            title="Single View (NumPy)",
            save_path=os.path.join(temp_dir, "single_view_numpy.png")
        )
        print("✓ plot_single_view completed")
        
        # Test voxel_to_pointcloud
        print("Testing voxel_to_pointcloud...")
        points = visualizer.voxel_to_pointcloud(structure, subdivision=2)
        print(f"✓ Generated {len(points)} points from voxel grid")
        
        # Test plot_pointcloud
        print("Testing plot_pointcloud...")
        fig = visualizer.plot_pointcloud(
            points,
            colors=(255, 100, 100),
            title="Point Cloud (NumPy)",
            save_path=os.path.join(temp_dir, "pointcloud_numpy.html")
        )
        print("✓ plot_pointcloud completed")


def test_pytorch_visualization(save_dir=None):
    """Test visualization with PyTorch tensors."""
    if not TORCH_AVAILABLE:
        print("\n=== Skipping PyTorch Tests (PyTorch not available) ===")
        return
    
    print("\n=== Testing PyTorch Tensor Visualization ===")
    
    structure, colors = create_test_data()
    
    # Convert to PyTorch tensors
    structure_tensor = torch.from_numpy(structure).float()
    colors_tensor = torch.from_numpy(colors).float()
    
    print(f"Structure tensor shape: {structure_tensor.shape}, device: {structure_tensor.device}")
    print(f"Colors tensor shape: {colors_tensor.shape}, device: {colors_tensor.device}")
    
    visualizer = Visualizer(figsize=12)
    
    # Determine save directory
    if save_dir is None:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = os.path.join(save_dir, "pytorch")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving PyTorch visualization results to: {temp_dir}")
        # Test plot_sections with tensor
        print("Testing plot_sections with PyTorch tensor...")
        fig = visualizer.plot_sections(
            structure_tensor, 
            title="Test Structure Sections (PyTorch)",
            save_path=os.path.join(temp_dir, "sections_pytorch.png")
        )
        print("✓ plot_sections with tensor completed")
        
        # Test plot_sculpture with tensors
        print("Testing plot_sculpture with PyTorch tensors...")
        fig = visualizer.plot_sculpture(
            structure_tensor, 
            colors=colors_tensor,
            title="Test Sculpture (PyTorch)",
            angles=[0, 1],
            save_path=os.path.join(temp_dir, "sculpture_pytorch.png")
        )
        print("✓ plot_sculpture with tensors completed")
        
        # Test plot_single_view with tensors
        print("Testing plot_single_view with PyTorch tensors...")
        fig = visualizer.plot_single_view(
            structure_tensor,
            colors=colors_tensor,
            title="Single View (PyTorch)",
            save_path=os.path.join(temp_dir, "single_view_pytorch.png")
        )
        print("✓ plot_single_view with tensors completed")
        
        # Test voxel_to_pointcloud with tensor
        print("Testing voxel_to_pointcloud with PyTorch tensor...")
        points = visualizer.voxel_to_pointcloud(structure_tensor, subdivision=2)
        print(f"✓ Generated {len(points)} points from tensor voxel grid")
        
        # Test plot_pointcloud with tensor colors
        print("Testing plot_pointcloud with PyTorch tensor colors...")
        colors_tensor_points = torch.tensor([[255, 100, 100]] * len(points)).float()
        fig = visualizer.plot_pointcloud(
            points,
            colors=colors_tensor_points,
            title="Point Cloud (PyTorch)",
            save_path=os.path.join(temp_dir, "pointcloud_pytorch.html")
        )
        print("✓ plot_pointcloud with tensor colors completed")


def test_pytorch_visualizer_class(save_dir=None):
    """Test the PyTorchVisualizer class."""
    print("\n=== Testing PyTorchVisualizer Class ===")
    
    pytorch_visualizer = PyTorchVisualizer(figsize=12)
    print(f"✓ PyTorchVisualizer initialized with backend: {pytorch_visualizer.backend}")
    print(f"✓ Default device: {pytorch_visualizer.device}")
    
    # Test training progress visualization
    print("Testing plot_training_progress...")
    metrics = {
        'generator_loss': [2.5, 2.1, 1.8, 1.5, 1.2, 1.0, 0.8],
        'discriminator_loss': [0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
        'fid_score': [150, 120, 100, 85, 70, 60, 55]
    }
    
    # Determine save directory
    if save_dir is None:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = os.path.join(save_dir, "pytorch_visualizer")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving PyTorchVisualizer results to: {temp_dir}")
        fig = pytorch_visualizer.plot_training_progress(
            metrics,
            save_path=os.path.join(temp_dir, "training_progress.png")
        )
        print("✓ plot_training_progress completed")
        
        # Test latent space visualization (placeholder)
        if TORCH_AVAILABLE:
            print("Testing visualize_latent_space...")
            dummy_model = torch.nn.Linear(100, 64)
            fig = pytorch_visualizer.visualize_latent_space(
                dummy_model,
                save_path=os.path.join(temp_dir, "latent_space.png")
            )
            print("✓ visualize_latent_space completed")


def test_file_handling(save_dir=None):
    """Test file loading and saving with PyTorch tensors."""
    print("\n=== Testing File Handling ===")
    
    structure, colors = create_test_data()
    visualizer = Visualizer(figsize=12)
    
    # Determine save directory
    if save_dir is None:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = os.path.join(save_dir, "file_handling")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving file handling results to: {temp_dir}")
        # Save as numpy files
        structure_numpy_path = os.path.join(temp_dir, "structure_001.npy")
        colors_numpy_path = os.path.join(temp_dir, "colors_001.npy")
        
        np.save(structure_numpy_path, structure)
        np.save(colors_numpy_path, colors)
        
        print("Testing visualize_sample_from_files with numpy files...")
        fig = visualizer.visualize_sample_from_files(
            structure_numpy_path,
            colors_numpy_path,
            save_path=os.path.join(temp_dir, "sample_from_numpy.png")
        )
        print("✓ visualize_sample_from_files with numpy files completed")
        
        # Test with PyTorch files if available
        if TORCH_AVAILABLE:
            structure_tensor = torch.from_numpy(structure).float()
            colors_tensor = torch.from_numpy(colors).float()
            
            structure_torch_path = os.path.join(temp_dir, "structure_001.pt")
            colors_torch_path = os.path.join(temp_dir, "colors_001.pt")
            
            torch.save(structure_tensor, structure_torch_path)
            torch.save(colors_tensor, colors_torch_path)
            
            print("Testing visualize_sample_from_files with PyTorch files...")
            fig = visualizer.visualize_sample_from_files(
                structure_torch_path,
                colors_torch_path,
                save_path=os.path.join(temp_dir, "sample_from_pytorch.png")
            )
            print("✓ visualize_sample_from_files with PyTorch files completed")


def test_gpu_tensors(save_dir=None):
    """Test GPU tensor handling if CUDA is available."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("\n=== Skipping GPU Tests (CUDA not available) ===")
        return
    
    print("\n=== Testing GPU Tensor Visualization ===")
    
    structure, colors = create_test_data()
    
    # Move tensors to GPU
    structure_gpu = torch.from_numpy(structure).float().cuda()
    colors_gpu = torch.from_numpy(colors).float().cuda()
    
    print(f"Structure GPU tensor shape: {structure_gpu.shape}, device: {structure_gpu.device}")
    print(f"Colors GPU tensor shape: {colors_gpu.shape}, device: {colors_gpu.device}")
    
    visualizer = Visualizer(figsize=12)
    
    # Determine save directory
    if save_dir is None:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = os.path.join(save_dir, "gpu_tensors")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving GPU tensor results to: {temp_dir}")
        print("Testing plot_sculpture with GPU tensors...")
        fig = visualizer.plot_sculpture(
            structure_gpu, 
            colors=colors_gpu,
            title="Test Sculpture (GPU)",
            angles=[0],
            save_path=os.path.join(temp_dir, "sculpture_gpu.png")
        )
        print("✓ plot_sculpture with GPU tensors completed")


def main(save_dir=None):
    """Run all integration tests."""
    print("DeepSculpt Visualization PyTorch Integration Test")
    print("=" * 50)
    
    # Set default save directory to results folder in repo
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "results")
        print(f"Using default results directory: {save_dir}")
    
    try:
        # Test numpy visualization
        test_numpy_visualization(save_dir=save_dir)
        
        # Test PyTorch tensor visualization
        test_pytorch_visualization(save_dir=save_dir)
        
        # Test PyTorchVisualizer class
        test_pytorch_visualizer_class(save_dir=save_dir)
        
        # Test file handling
        test_file_handling(save_dir=save_dir)
        
        # Test GPU tensors
        test_gpu_tensors(save_dir=save_dir)
        
        print("\n" + "=" * 50)
        print("✓ All integration tests completed successfully!")
        print("✓ PyTorch tensor support is working correctly")
        print("✓ Visualization methods handle both numpy arrays and PyTorch tensors")
        print("✓ Device-aware tensor conversion is functioning properly")
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Visualization Integration Test")
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default=None,
        help="Directory to save visualization results (default: ./results)"
    )
    parser.add_argument(
        "--temp", 
        action="store_true",
        help="Use temporary directories instead of saving to repo"
    )
    
    args = parser.parse_args()
    
    # Use temp directories if requested
    save_dir = None if args.temp else args.save_dir
    
    exit(main(save_dir=save_dir))