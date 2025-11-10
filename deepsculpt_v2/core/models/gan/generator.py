"""
GAN Generator models for DeepSculpt PyTorch implementation.

This module contains all generator architectures for 3D GAN models,
including simple, complex, skip, monochrome, autoencoder, progressive,
and conditional generators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math

from ..base_models import BaseGenerator, SparseConv3d, SparseConvTranspose3d, SparseBatchNorm3d


class SimpleGenerator(BaseGenerator):
    """Simple generator model equivalent to TensorFlow version."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        # Initial dense layer
        self.initial_size = void_dim // 8
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.relu = nn.ReLU()
        self.threshold_relu = nn.Threshold(0.0, 0.0)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        # Handle single batch case for batch norm
        if x.size(0) == 1 and self.training:
            self.bn1.eval()
            x = self.bn1(x)
            self.bn1.train()
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)
        
        # First transposed conv block
        x = self.conv1(x)
        if x.size(0) == 1 and self.training:
            self.bn2.eval()
            x = self.bn2(x)
            self.bn2.train()
        else:
            x = self.bn2(x)
        x = self.relu(x)
        
        # Second transposed conv block
        x = self.conv2(x)
        if x.size(0) == 1 and self.training:
            self.bn3.eval()
            x = self.bn3(x)
            self.bn3.train()
        else:
            x = self.bn3(x)
        x = self.relu(x)
        
        # Third transposed conv block
        x = self.conv3(x)
        if x.size(0) == 1 and self.training:
            self.bn4.eval()
            x = self.bn4(x)
            self.bn4.train()
        else:
            x = self.bn4(x)
        x = self.relu(x)
        
        # Final transposed conv block
        x = self.conv4(x)
        x = self.softmax(x)
        x = self.threshold_relu(x)
        
        # Output is already in PyTorch format: (batch, channels, D, H, W)
        # No need to reshape - conv4 already outputs the correct shape
        
        return x


class ComplexGenerator(BaseGenerator):
    """Complex generator model with skip connections."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        self.initial_size = void_dim // 8
        
        # Initial dense layer
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers with skip connections
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        # Skip connection layer - concatenates with previous layer
        self.conv2 = ConvTranspose(noise_dim * 2, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2 * 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4 * 2, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)
        
        skip_connections = []
        
        # First layer
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        skip_connections.append(x)
        
        # Second layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        skip_connections.append(x)
        
        # Third layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        skip_connections.append(x)
        
        # Final layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv4(x)
        x = self.tanh(x)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class SkipGenerator(BaseGenerator):
    """Generator with skip connections (U-Net style)."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        self.initial_size = void_dim // 8
        
        # Initial dense layer
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim * 2, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2 * 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4 * 2, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.relu = nn.ReLU()
        self.threshold_relu = nn.Threshold(0.0, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)
        
        skip_connections = []
        
        # First layer
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        skip_connections.append(x)
        
        # Second layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        skip_connections.append(x)
        
        # Third layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        skip_connections.append(x)
        
        # Final layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv4(x)
        x = self.threshold_relu(x)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class MonochromeGenerator(BaseGenerator):
    """Monochrome generator model."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 0, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        # Always 3 channels for monochrome
        self.output_channels = 3
        self.initial_size = void_dim // 8
        
        # Initial dense layer
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.relu = nn.ReLU()
        self.threshold_relu = nn.Threshold(0.0, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)
        
        # First transposed conv block
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Second transposed conv block
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Third transposed conv block
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Final transposed conv block
        x = self.conv4(x)
        x = self.relu(x)
        x = self.threshold_relu(x)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class AutoencoderGenerator(BaseGenerator):
    """Generator based on autoencoder architecture."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        # Dense layer to expand the latent dimension
        self.fc = nn.Linear(noise_dim, 4 * 4 * 4 * 16)  # 1024 values for 4x4x4x16
        
        # Upsampling layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        
        self.conv1 = ConvTranspose(16, 128, 5, 2, 2, 1)
        self.conv2 = ConvTranspose(128, 64, 5, 2, 2, 1)
        self.conv3 = ConvTranspose(64, self.output_channels, 5, 2, 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense layer to expand latent dimension
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(-1, 16, 4, 4, 4)  # Reshape to 4x4x4x16
        
        # Upsampling layers
        x = self.conv1(x)  # 4x4x4 -> 8x8x8
        x = self.relu(x)
        
        x = self.conv2(x)  # 8x8x8 -> 16x16x16
        x = self.relu(x)
        
        x = self.conv3(x)  # 16x16x16 -> 32x32x32
        x = self.sigmoid(x)
        
        # Reshape to match expected format (batch, depth, height, width, channels)
        x = x.permute(0, 2, 3, 4, 1)
        
        return x


class ProgressiveGenerator(BaseGenerator):
    """Progressive growing generator for high-resolution 3D data."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, 
                 max_resolution: int = 128, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        self.max_resolution = max_resolution
        
        # Progressive layers for different resolutions
        self.initial_block = self._make_initial_block()
        self.progressive_blocks = nn.ModuleList()
        
        # Create progressive blocks for each resolution level
        current_res = 8
        current_channels = noise_dim
        
        while current_res <= max_resolution:
            block = self._make_progressive_block(current_channels, current_channels // 2)
            self.progressive_blocks.append(block)
            current_channels = current_channels // 2
            current_res *= 2
        
        # Final output layer
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        self.to_rgb = ConvTranspose(current_channels, self.output_channels, 1, 1, 0)
        
        self.current_level = 0  # Current progressive level
        self.alpha = 1.0  # Blending factor for progressive growing
    
    def _make_initial_block(self):
        """Create the initial 4x4x4 block."""
        ConvTranspose = SparseConvTranspose3d if self.sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            nn.Linear(self.noise_dim, 4 * 4 * 4 * self.noise_dim),
            nn.Unflatten(1, (self.noise_dim, 4, 4, 4)),
            ConvTranspose(self.noise_dim, self.noise_dim, 3, 1, 1),
            BatchNorm(self.noise_dim),
            nn.LeakyReLU(0.2)
        )
    
    def _make_progressive_block(self, in_channels: int, out_channels: int):
        """Create a progressive block that doubles the resolution."""
        ConvTranspose = SparseConvTranspose3d if self.sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvTranspose(in_channels, out_channels, 3, 1, 1),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2),
            ConvTranspose(out_channels, out_channels, 3, 1, 1),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial block
        x = self.initial_block(x)
        
        # Progressive blocks up to current level
        for i in range(min(self.current_level + 1, len(self.progressive_blocks))):
            x = self.progressive_blocks[i](x)
        
        # Convert to RGB
        x = self.to_rgb(x)
        
        return x
    
    def grow(self):
        """Grow the network by one level."""
        if self.current_level < len(self.progressive_blocks) - 1:
            self.current_level += 1
            self.alpha = 0.0  # Start with full blend to new layer
    
    def set_alpha(self, alpha: float):
        """Set the blending factor for progressive growing."""
        self.alpha = max(0.0, min(1.0, alpha))


class ConditionalGenerator(BaseGenerator):
    """Conditional generator for controlled generation."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, condition_dim: int = 10,
                 color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        self.condition_dim = condition_dim
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, noise_dim)
        
        # Combined input dimension
        combined_dim = noise_dim + noise_dim  # noise + embedded condition
        
        self.initial_size = void_dim // 8
        
        # Initial dense layer with combined input
        self.fc = nn.Linear(combined_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            # Embed condition
            condition_embedded = self.condition_embedding(condition)
            # Concatenate noise and condition
            x = torch.cat([x, condition_embedded], dim=1)
        
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)
        
        # Transposed conv blocks
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.tanh(x)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x