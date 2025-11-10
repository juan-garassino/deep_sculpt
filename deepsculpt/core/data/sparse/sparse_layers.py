#!/usr/bin/env python3
"""
Sparse Tensor Neural Network Layers for DeepSculpt v2.0

Implements sparse-aware neural network layers for memory-efficient 3D processing:
- SparseConv3d: Sparse 3D convolution layer
- SparseConvTranspose3d: Sparse 3D transposed convolution
- SparseBatchNorm3d: Sparse batch normalization
- SparseActivation: Sparse-aware activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import warnings


class SparseConv3d(nn.Module):
    """Sparse-aware 3D convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], 
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1, bias: bool = True,
                 sparse_threshold: float = 0.1):
        super().__init__()
        
        self.sparse_threshold = sparse_threshold
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, 
                               stride, padding, dilation, groups, bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse tensor optimization."""
        # Check if input is sparse or should be converted
        if hasattr(x, 'is_sparse') and x.is_sparse:
            # Handle sparse tensor input
            x_dense = x.to_dense()
            output = self.conv3d(x_dense)
        else:
            output = self.conv3d(x)
        
        # Convert output to sparse if beneficial
        sparsity = (output == 0).float().mean()
        if sparsity > self.sparse_threshold:
            try:
                output = output.to_sparse()
            except:
                # Fallback to dense if sparse conversion fails
                pass
        
        return output


class SparseConvTranspose3d(nn.Module):
    """Sparse-aware 3D transposed convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int, int]] = 0,
                 groups: int = 1, bias: bool = True, dilation: Union[int, Tuple[int, int, int]] = 1,
                 sparse_threshold: float = 0.1):
        super().__init__()
        
        self.sparse_threshold = sparse_threshold
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride, padding, output_padding, groups, bias, dilation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse tensor optimization."""
        # Handle sparse input
        if hasattr(x, 'is_sparse') and x.is_sparse:
            x_dense = x.to_dense()
            output = self.conv_transpose3d(x_dense)
        else:
            output = self.conv_transpose3d(x)
        
        # Convert to sparse if beneficial
        sparsity = (output == 0).float().mean()
        if sparsity > self.sparse_threshold:
            try:
                output = output.to_sparse()
            except:
                pass
        
        return output


class SparseBatchNorm3d(nn.Module):
    """Sparse-aware 3D batch normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        
        self.batch_norm3d = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass handling sparse tensors."""
        if hasattr(x, 'is_sparse') and x.is_sparse:
            # Convert to dense for batch norm, then back to sparse
            x_dense = x.to_dense()
            output = self.batch_norm3d(x_dense)
            
            # Convert back to sparse if input was sparse
            try:
                output = output.to_sparse()
            except:
                pass
            
            return output
        else:
            return self.batch_norm3d(x)


class SparseActivation(nn.Module):
    """Sparse-aware activation function wrapper."""
    
    def __init__(self, activation: nn.Module, sparse_threshold: float = 0.1):
        super().__init__()
        self.activation = activation
        self.sparse_threshold = sparse_threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation with sparse tensor handling."""
        if hasattr(x, 'is_sparse') and x.is_sparse:
            x_dense = x.to_dense()
            output = self.activation(x_dense)
            
            # Convert back to sparse if beneficial
            sparsity = (output == 0).float().mean()
            if sparsity > self.sparse_threshold:
                try:
                    output = output.to_sparse()
                except:
                    pass
            
            return output
        else:
            output = self.activation(x)
            
            # Convert to sparse if beneficial
            sparsity = (output == 0).float().mean()
            if sparsity > self.sparse_threshold:
                try:
                    output = output.to_sparse()
                except:
                    pass
            
            return output


# Convenience functions for creating sparse layers
def sparse_conv3d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], **kwargs):
    """Create a sparse-aware 3D convolution layer."""
    return SparseConv3d(in_channels, out_channels, kernel_size, **kwargs)


def sparse_conv_transpose3d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], **kwargs):
    """Create a sparse-aware 3D transposed convolution layer."""
    return SparseConvTranspose3d(in_channels, out_channels, kernel_size, **kwargs)


def sparse_batch_norm3d(num_features: int, **kwargs):
    """Create a sparse-aware 3D batch normalization layer."""
    return SparseBatchNorm3d(num_features, **kwargs)


def sparse_relu(sparse_threshold: float = 0.1):
    """Create a sparse-aware ReLU activation."""
    return SparseActivation(nn.ReLU(inplace=True), sparse_threshold)


def sparse_leaky_relu(negative_slope: float = 0.01, sparse_threshold: float = 0.1):
    """Create a sparse-aware LeakyReLU activation."""
    return SparseActivation(nn.LeakyReLU(negative_slope, inplace=True), sparse_threshold)