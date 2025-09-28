"""
Test suite for sparse tensor implementations in the DeepSculpt PyTorch migration.

This test suite validates:
1. Sparse tensor detection and conversion utilities
2. Sparse-aware neural network layers
3. Sparse 3D convolution layers
4. Sparse normalization and activation layers
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

# Import the sparse tensor utilities and layers
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from pytorch_utils import SparseTensorHandler, MemoryOptimizer, batch_sparse_conversion
from pytorch_models import (
    SparseConv3d, SparseConvTranspose3d, SparseBatchNorm3d,
    AdvancedSparseConv3d, AdvancedSparseConvTranspose3d, AdvancedSparseBatchNorm3d,
    SparseActivation, AdvancedSparseActivation, SparseDropout3d,
    SparsePooling3d, SparseUpsampling3d, SparseLinear,
    SparseGroupNorm3d, SparseInstanceNorm3d,
    convert_model_to_sparse, analyze_model_sparsity
)


class TestSparseTensorHandler:
    """Test the SparseTensorHandler utility class."""
    
    def test_sparsity_detection(self):
        """Test sparsity detection for dense and sparse tensors."""
        # Create a dense tensor with known sparsity
        dense_tensor = torch.zeros(10, 10, 10)
        dense_tensor[0:2, 0:2, 0:2] = 1.0  # 8/1000 = 0.008 non-zero ratio
        
        sparsity = SparseTensorHandler.detect_sparsity(dense_tensor)
        expected_sparsity = 992 / 1000  # 992 zeros out of 1000 elements
        assert abs(sparsity - expected_sparsity) < 0.01
        
        # Test sparse tensor
        sparse_tensor = dense_tensor.to_sparse()
        sparse_sparsity = SparseTensorHandler.detect_sparsity(sparse_tensor)
        assert abs(sparse_sparsity - expected_sparsity) < 0.01
    
    def test_sparse_conversion(self):
        """Test conversion between dense and sparse representations."""
        # Create a sparse tensor
        dense_tensor = torch.zeros(8, 8, 8)
        dense_tensor[0:2, 0:2, 0:2] = torch.randn(2, 2, 2)
        
        # Convert to sparse
        sparse_tensor = SparseTensorHandler.to_sparse(dense_tensor)
        assert sparse_tensor.is_sparse
        assert sparse_tensor._nnz() == 8  # 2x2x2 non-zero elements
        
        # Convert back to dense
        reconstructed_dense = SparseTensorHandler.to_dense(sparse_tensor)
        assert not reconstructed_dense.is_sparse
        assert torch.allclose(dense_tensor, reconstructed_dense)
    
    def test_auto_conversion(self):
        """Test automatic sparse/dense conversion based on sparsity."""
        # Very sparse tensor - should convert to sparse
        very_sparse = torch.zeros(16, 16, 16)
        very_sparse[0, 0, 0] = 1.0
        
        result = SparseTensorHandler.auto_convert(very_sparse, sparsity_threshold=0.5)
        assert result.is_sparse
        
        # Dense tensor - should remain dense
        dense = torch.randn(4, 4, 4)
        result = SparseTensorHandler.auto_convert(dense, sparsity_threshold=0.5)
        assert not result.is_sparse
    
    def test_sparse_info(self):
        """Test sparse tensor information extraction."""
        tensor = torch.zeros(8, 8, 8)
        tensor[0:2, 0:2, 0:2] = 1.0
        
        info = SparseTensorHandler.get_sparse_info(tensor)
        
        assert info["is_sparse"] == False
        assert info["total_elements"] == 512
        assert info["sparsity"] > 0.9  # Most elements are zero
        
        sparse_tensor = tensor.to_sparse()
        sparse_info = SparseTensorHandler.get_sparse_info(sparse_tensor)
        
        assert sparse_info["is_sparse"] == True
        assert sparse_info["stored_values"] == 8
    
    def test_representation_comparison(self):
        """Test comparison between sparse and dense representations."""
        tensor = torch.zeros(16, 16, 16)
        tensor[0:2, 0:2, 0:2] = torch.randn(2, 2, 2)
        
        comparison = SparseTensorHandler.compare_representations(tensor)
        
        assert "dense" in comparison
        assert "sparse" in comparison
        assert "memory_savings" in comparison
        assert comparison["memory_savings"]["percentage"] > 0  # Should save memory


class TestSparseConvolutionLayers:
    """Test sparse convolution layer implementations."""
    
    def test_sparse_conv3d_basic(self):
        """Test basic SparseConv3d functionality."""
        layer = SparseConv3d(1, 2, kernel_size=3, padding=1)
        
        # Test with dense input
        dense_input = torch.randn(1, 1, 8, 8, 8)
        dense_output = layer(dense_input)
        assert dense_output.shape == (1, 2, 8, 8, 8)
        
        # Test with sparse input
        sparse_input = dense_input.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (1, 2, 8, 8, 8)
    
    def test_advanced_sparse_conv3d(self):
        """Test AdvancedSparseConv3d with different algorithms."""
        layer = AdvancedSparseConv3d(1, 2, kernel_size=3, padding=1, sparse_algorithm="auto")
        
        # Create a sparse input
        input_tensor = torch.zeros(1, 1, 8, 8, 8)
        input_tensor[0, 0, 0:2, 0:2, 0:2] = torch.randn(2, 2, 2)
        sparse_input = input_tensor.to_sparse()
        
        output = layer(sparse_input)
        assert output.shape == (1, 2, 8, 8, 8)
        
        # Check performance stats
        stats = layer.get_performance_stats()
        assert "sparse_ops_ratio" in stats
        assert "total_operations" in stats
    
    def test_sparse_conv_transpose3d(self):
        """Test SparseConvTranspose3d functionality."""
        layer = SparseConvTranspose3d(2, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        input_tensor = torch.randn(1, 2, 4, 4, 4)
        output = layer(input_tensor)
        assert output.shape == (1, 1, 8, 8, 8)
        
        # Test with sparse input
        sparse_input = input_tensor.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (1, 1, 8, 8, 8)
    
    def test_sparse_separable_conv3d(self):
        """Test SparseSeparableConv3d functionality."""
        from pytorch_models import SparseSeparableConv3d
        
        layer = SparseSeparableConv3d(4, 8, kernel_size=3, padding=1)
        
        input_tensor = torch.randn(1, 4, 8, 8, 8)
        output = layer(input_tensor)
        assert output.shape == (1, 8, 8, 8, 8)
        
        # Check performance stats
        stats = layer.get_performance_stats()
        assert "depthwise_stats" in stats
        assert "pointwise_stats" in stats


class TestSparseNormalizationLayers:
    """Test sparse normalization layer implementations."""
    
    def test_sparse_batch_norm3d(self):
        """Test SparseBatchNorm3d functionality."""
        layer = SparseBatchNorm3d(4)
        
        # Test with dense input
        dense_input = torch.randn(2, 4, 8, 8, 8)
        dense_output = layer(dense_input)
        assert dense_output.shape == (2, 4, 8, 8, 8)
        
        # Test with sparse input
        sparse_input = dense_input.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (2, 4, 8, 8, 8)
    
    def test_advanced_sparse_batch_norm3d(self):
        """Test AdvancedSparseBatchNorm3d with different modes."""
        layer = AdvancedSparseBatchNorm3d(4, sparse_mode="adaptive")
        
        # Create sparse input
        input_tensor = torch.zeros(2, 4, 8, 8, 8)
        input_tensor[:, :, 0:2, 0:2, 0:2] = torch.randn(2, 4, 2, 2, 2)
        sparse_input = input_tensor.to_sparse()
        
        output = layer(sparse_input)
        assert output.shape == (2, 4, 8, 8, 8)
        
        # Check normalization stats
        stats = layer.get_normalization_stats()
        assert "mode_usage_ratios" in stats
        assert "total_operations" in stats
    
    def test_sparse_group_norm3d(self):
        """Test SparseGroupNorm3d functionality."""
        layer = SparseGroupNorm3d(num_groups=2, num_channels=4)
        
        input_tensor = torch.randn(1, 4, 8, 8, 8)
        output = layer(input_tensor)
        assert output.shape == (1, 4, 8, 8, 8)
        
        # Test with sparse input
        sparse_input = input_tensor.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (1, 4, 8, 8, 8)
    
    def test_sparse_instance_norm3d(self):
        """Test SparseInstanceNorm3d functionality."""
        layer = SparseInstanceNorm3d(4)
        
        input_tensor = torch.randn(2, 4, 8, 8, 8)
        output = layer(input_tensor)
        assert output.shape == (2, 4, 8, 8, 8)


class TestSparseActivationLayers:
    """Test sparse activation layer implementations."""
    
    def test_sparse_activation_basic(self):
        """Test basic SparseActivation functionality."""
        layer = SparseActivation("relu")
        
        # Test with dense input
        dense_input = torch.randn(1, 4, 8, 8, 8)
        dense_output = layer(dense_input)
        assert dense_output.shape == (1, 4, 8, 8, 8)
        assert torch.all(dense_output >= 0)  # ReLU property
        
        # Test with sparse input
        sparse_input = dense_input.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (1, 4, 8, 8, 8)
    
    def test_advanced_sparse_activation(self):
        """Test AdvancedSparseActivation with different functions."""
        activations = ["relu", "leaky_relu", "gelu", "swish", "tanh"]
        
        for activation in activations:
            layer = AdvancedSparseActivation(activation, adaptive_threshold=True)
            
            input_tensor = torch.randn(1, 2, 4, 4, 4)
            output = layer(input_tensor)
            assert output.shape == (1, 2, 4, 4, 4)
            
            # Check activation stats
            stats = layer.get_activation_stats()
            assert "activation_type" in stats
            assert stats["activation_type"] == activation
    
    def test_sparse_dropout3d(self):
        """Test SparseDropout3d functionality."""
        layer = SparseDropout3d(p=0.5, sparse_aware=True)
        layer.train()  # Enable dropout
        
        input_tensor = torch.ones(1, 4, 8, 8, 8)
        sparse_input = input_tensor.to_sparse()
        
        output = layer(sparse_input)
        assert output.shape == (1, 4, 8, 8, 8)
        assert output.is_sparse


class TestSparseUtilityLayers:
    """Test sparse utility layer implementations."""
    
    def test_sparse_pooling3d(self):
        """Test SparsePooling3d functionality."""
        layer = SparsePooling3d(kernel_size=2, stride=2, pool_type="max")
        
        input_tensor = torch.randn(1, 4, 8, 8, 8)
        output = layer(input_tensor)
        assert output.shape == (1, 4, 4, 4, 4)
        
        # Test with sparse input
        sparse_input = input_tensor.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (1, 4, 4, 4, 4)
    
    def test_sparse_upsampling3d(self):
        """Test SparseUpsampling3d functionality."""
        layer = SparseUpsampling3d(scale_factor=2, mode="nearest")
        
        input_tensor = torch.randn(1, 4, 4, 4, 4)
        output = layer(input_tensor)
        assert output.shape == (1, 4, 8, 8, 8)
        
        # Test with sparse input
        sparse_input = input_tensor.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (1, 4, 8, 8, 8)
    
    def test_sparse_linear(self):
        """Test SparseLinear functionality."""
        layer = SparseLinear(64, 32)
        
        input_tensor = torch.randn(2, 64)
        output = layer(input_tensor)
        assert output.shape == (2, 32)
        
        # Test with sparse input
        sparse_input = input_tensor.to_sparse()
        sparse_output = layer(sparse_input)
        assert sparse_output.shape == (2, 32)


class TestSparseModelUtilities:
    """Test sparse model utility functions."""
    
    def test_convert_model_to_sparse(self):
        """Test converting a regular model to use sparse layers."""
        # Create a simple model
        model = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 2, 3, padding=1)
        )
        
        # Convert to sparse
        sparse_model = convert_model_to_sparse(model, sparse_threshold=0.1)
        
        # Test that it still works
        input_tensor = torch.randn(1, 1, 8, 8, 8)
        output = sparse_model(input_tensor)
        assert output.shape == (1, 2, 8, 8, 8)
    
    def test_analyze_model_sparsity(self):
        """Test sparsity analysis of a model."""
        model = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 2, 3, padding=1)
        )
        
        input_tensor = torch.randn(1, 1, 8, 8, 8)
        analysis = analyze_model_sparsity(model, input_tensor)
        
        assert "layer_sparsities" in analysis
        assert "average_sparsity" in analysis
        assert "num_layers_analyzed" in analysis
        assert analysis["num_layers_analyzed"] > 0


class TestBatchSparseOperations:
    """Test batch sparse tensor operations."""
    
    def test_batch_sparse_conversion(self):
        """Test batch conversion of tensors to optimal representations."""
        # Create a mix of sparse and dense tensors
        tensors = [
            torch.zeros(8, 8, 8),  # Very sparse
            torch.randn(4, 4, 4),  # Dense
            torch.zeros(16, 16, 16)  # Very sparse
        ]
        
        # Add some non-zero values to sparse tensors
        tensors[0][0:2, 0:2, 0:2] = torch.randn(2, 2, 2)
        tensors[2][0:1, 0:1, 0:1] = torch.randn(1, 1, 1)
        
        converted = batch_sparse_conversion(tensors, sparsity_threshold=0.5)
        
        assert len(converted) == 3
        assert converted[0].is_sparse  # Should be converted to sparse
        assert not converted[1].is_sparse  # Should remain dense
        assert converted[2].is_sparse  # Should be converted to sparse


class TestMemoryOptimization:
    """Test memory optimization utilities."""
    
    def test_memory_usage_calculation(self):
        """Test memory usage calculation for tensors."""
        dense_tensor = torch.randn(16, 16, 16)
        sparse_tensor = dense_tensor.to_sparse()
        
        dense_memory = MemoryOptimizer.get_tensor_memory_usage(dense_tensor)
        sparse_memory = MemoryOptimizer.get_tensor_memory_usage(sparse_tensor)
        
        assert dense_memory > 0
        assert sparse_memory > 0
        # For a dense tensor converted to sparse, sparse should use more memory
        # due to indices overhead
    
    def test_memory_optimization_suggestions(self):
        """Test memory optimization suggestions."""
        # Create a very sparse tensor
        tensor = torch.zeros(32, 32, 32)
        tensor[0:2, 0:2, 0:2] = torch.randn(2, 2, 2)
        
        suggestions = MemoryOptimizer.suggest_optimization(tensor)
        
        assert "current_memory" in suggestions
        assert "sparsity" in suggestions
        assert "suggestions" in suggestions
        assert len(suggestions["suggestions"]) > 0


class TestIntegration:
    """Integration tests for sparse tensor functionality."""
    
    def test_end_to_end_sparse_model(self):
        """Test a complete sparse model end-to-end."""
        from pytorch_models import SparseConvBlock3d, SparseSequential
        
        # Create a model with sparse layers
        model = SparseSequential(
            SparseConvBlock3d(1, 4, kernel_size=3, padding=1),
            SparseConvBlock3d(4, 8, kernel_size=3, padding=1),
            SparsePooling3d(kernel_size=2, stride=2),
            SparseConvBlock3d(8, 4, kernel_size=3, padding=1),
            SparseUpsampling3d(scale_factor=2),
            SparseConvBlock3d(4, 1, kernel_size=3, padding=1)
        )
        
        # Test with both dense and sparse inputs
        dense_input = torch.randn(1, 1, 8, 8, 8)
        dense_output = model(dense_input)
        assert dense_output.shape == (1, 1, 8, 8, 8)
        
        # Create sparse input
        sparse_input_data = torch.zeros(1, 1, 8, 8, 8)
        sparse_input_data[0, 0, 0:4, 0:4, 0:4] = torch.randn(4, 4, 4)
        sparse_input = sparse_input_data.to_sparse()
        
        sparse_output = model(sparse_input)
        assert sparse_output.shape == (1, 1, 8, 8, 8)
        
        # Check sparsity profile
        sparsity_profile = model.get_sparsity_profile()
        assert len(sparsity_profile) > 0
    
    def test_sparse_model_training_compatibility(self):
        """Test that sparse models are compatible with training."""
        model = nn.Sequential(
            SparseConv3d(1, 4, 3, padding=1),
            SparseBatchNorm3d(4),
            SparseActivation("relu"),
            SparseConv3d(4, 1, 3, padding=1)
        )
        
        # Create training data
        input_data = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn(2, 1, 8, 8, 8)
        
        # Forward pass
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        assert input_data.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    # Run basic tests
    print("Running sparse tensor implementation tests...")
    
    # Test SparseTensorHandler
    handler_tests = TestSparseTensorHandler()
    handler_tests.test_sparsity_detection()
    handler_tests.test_sparse_conversion()
    handler_tests.test_auto_conversion()
    handler_tests.test_sparse_info()
    handler_tests.test_representation_comparison()
    print("✓ SparseTensorHandler tests passed")
    
    # Test sparse convolution layers
    conv_tests = TestSparseConvolutionLayers()
    conv_tests.test_sparse_conv3d_basic()
    conv_tests.test_advanced_sparse_conv3d()
    conv_tests.test_sparse_conv_transpose3d()
    print("✓ Sparse convolution layer tests passed")
    
    # Test sparse normalization layers
    norm_tests = TestSparseNormalizationLayers()
    norm_tests.test_sparse_batch_norm3d()
    norm_tests.test_advanced_sparse_batch_norm3d()
    norm_tests.test_sparse_group_norm3d()
    norm_tests.test_sparse_instance_norm3d()
    print("✓ Sparse normalization layer tests passed")
    
    # Test sparse activation layers
    activation_tests = TestSparseActivationLayers()
    activation_tests.test_sparse_activation_basic()
    activation_tests.test_advanced_sparse_activation()
    activation_tests.test_sparse_dropout3d()
    print("✓ Sparse activation layer tests passed")
    
    # Test utility layers
    utility_tests = TestSparseUtilityLayers()
    utility_tests.test_sparse_pooling3d()
    utility_tests.test_sparse_upsampling3d()
    utility_tests.test_sparse_linear()
    print("✓ Sparse utility layer tests passed")
    
    # Test model utilities
    model_tests = TestSparseModelUtilities()
    model_tests.test_convert_model_to_sparse()
    model_tests.test_analyze_model_sparsity()
    print("✓ Sparse model utility tests passed")
    
    # Test batch operations
    batch_tests = TestBatchSparseOperations()
    batch_tests.test_batch_sparse_conversion()
    print("✓ Batch sparse operation tests passed")
    
    # Test memory optimization
    memory_tests = TestMemoryOptimization()
    memory_tests.test_memory_usage_calculation()
    memory_tests.test_memory_optimization_suggestions()
    print("✓ Memory optimization tests passed")
    
    # Test integration
    integration_tests = TestIntegration()
    integration_tests.test_end_to_end_sparse_model()
    integration_tests.test_sparse_model_training_compatibility()
    print("✓ Integration tests passed")
    
    print("\n🎉 All sparse tensor implementation tests passed successfully!")
    print("\nImplemented features:")
    print("- Sparse tensor detection and conversion utilities")
    print("- Advanced sparse 3D convolution layers")
    print("- Sparse normalization layers (BatchNorm, GroupNorm, InstanceNorm)")
    print("- Sparse activation layers with multiple activation functions")
    print("- Sparse utility layers (Pooling, Upsampling, Linear, Dropout)")
    print("- Model conversion utilities for sparse optimization")
    print("- Memory optimization and profiling tools")
    print("- Comprehensive sparsity analysis and benchmarking")