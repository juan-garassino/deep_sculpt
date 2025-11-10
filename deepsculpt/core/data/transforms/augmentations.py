"""
Data augmentation pipeline for DeepSculpt PyTorch implementation.

This module provides comprehensive augmentation techniques specifically
designed for 3D sculpture data including geometric transformations,
noise injection, and domain-specific augmentations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import random
import math
from abc import ABC, abstractmethod


class BaseAugmentation(ABC):
    """
    Abstract base class for all augmentations.
    """
    
    def __init__(self, probability: float = 0.5, device: str = "cuda"):
        """
        Initialize base augmentation.
        
        Args:
            probability: Probability of applying this augmentation
            device: Device for tensor operations
        """
        self.probability = probability
        self.device = device
    
    @abstractmethod
    def apply(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Augmented data dictionary
        """
        pass
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation with probability check."""
        if random.random() < self.probability:
            return self.apply(data)
        return data
    
    def to(self, device: str):
        """Move augmentation to device."""
        self.device = device
        return self


class RotationAugmentation(BaseAugmentation):
    """
    3D rotation augmentation for sculpture data.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-30.0, 30.0),
        axes: List[str] = ["x", "y", "z"],
        probability: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize rotation augmentation.
        
        Args:
            rotation_range: Range of rotation angles in degrees
            axes: List of axes to rotate around ("x", "y", "z")
            probability: Probability of applying augmentation
            device: Device for tensor operations
        """
        super().__init__(probability, device)
        self.rotation_range = rotation_range
        self.axes = axes
    
    def apply(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply rotation augmentation."""
        result = {}
        
        # Generate random rotation angles
        angles = {}
        for axis in self.axes:
            angles[axis] = random.uniform(*self.rotation_range)
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"] and len(tensor.shape) >= 3:
                result[key] = self._rotate_tensor(tensor.to(self.device), angles)
            else:
                result[key] = tensor
        
        return result
    
    def _rotate_tensor(self, tensor: torch.Tensor, angles: Dict[str, float]) -> torch.Tensor:
        """Rotate tensor using 3D rotation matrices."""
        # Convert angles to radians
        angles_rad = {axis: math.radians(angle) for axis, angle in angles.items()}
        
        # Create rotation matrices
        rotation_matrices = []
        
        if "x" in angles_rad:
            cos_x, sin_x = math.cos(angles_rad["x"]), math.sin(angles_rad["x"])
            rx = torch.tensor([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], dtype=tensor.dtype, device=self.device)
            rotation_matrices.append(rx)
        
        if "y" in angles_rad:
            cos_y, sin_y = math.cos(angles_rad["y"]), math.sin(angles_rad["y"])
            ry = torch.tensor([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ], dtype=tensor.dtype, device=self.device)
            rotation_matrices.append(ry)
        
        if "z" in angles_rad:
            cos_z, sin_z = math.cos(angles_rad["z"]), math.sin(angles_rad["z"])
            rz = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], dtype=tensor.dtype, device=self.device)
            rotation_matrices.append(rz)
        
        # Combine rotation matrices
        combined_rotation = torch.eye(3, device=self.device, dtype=tensor.dtype)
        for rm in rotation_matrices:
            combined_rotation = torch.mm(combined_rotation, rm)
        
        # Apply rotation using affine transformation
        return self._apply_affine_transform(tensor, combined_rotation)
    
    def _apply_affine_transform(self, tensor: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation to 3D tensor."""
        # Create affine transformation matrix
        affine_matrix = torch.zeros(3, 4, device=self.device, dtype=tensor.dtype)
        affine_matrix[:3, :3] = rotation_matrix
        
        # Handle different tensor shapes
        if len(tensor.shape) == 3:  # (D, H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            squeeze_dims = [0, 1]
        elif len(tensor.shape) == 4:  # (D, H, W, C) or (B, D, H, W)
            if tensor.shape[-1] <= 10:  # Assume last dim is channels
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
                squeeze_dims = [0]
                permute_back = True
            else:
                tensor = tensor.unsqueeze(1)  # (B, 1, D, H, W)
                squeeze_dims = [1]
                permute_back = False
        elif len(tensor.shape) == 5:  # (B, D, H, W, C)
            tensor = tensor.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
            permute_back = True
        else:
            return tensor  # Unsupported shape
        
        # Create grid and apply transformation
        grid = F.affine_grid(
            affine_matrix.unsqueeze(0).expand(tensor.shape[0], -1, -1),
            tensor.shape,
            align_corners=False
        )
        
        # Apply transformation
        transformed = F.grid_sample(
            tensor,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        # Restore original shape
        for dim in reversed(squeeze_dims):
            transformed = transformed.squeeze(dim)
        
        if 'permute_back' in locals() and permute_back:
            if len(transformed.shape) == 4:  # (C, D, H, W) -> (D, H, W, C)
                transformed = transformed.permute(1, 2, 3, 0)
            elif len(transformed.shape) == 5:  # (B, C, D, H, W) -> (B, D, H, W, C)
                transformed = transformed.permute(0, 2, 3, 4, 1)
        
        return transformed


class ScalingAugmentation(BaseAugmentation):
    """
    3D scaling augmentation for sculpture data.
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        uniform_scaling: bool = True,
        probability: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize scaling augmentation.
        
        Args:
            scale_range: Range of scaling factors
            uniform_scaling: Whether to use uniform scaling for all dimensions
            probability: Probability of applying augmentation
            device: Device for tensor operations
        """
        super().__init__(probability, device)
        self.scale_range = scale_range
        self.uniform_scaling = uniform_scaling
    
    def apply(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply scaling augmentation."""
        result = {}
        
        # Generate scaling factors
        if self.uniform_scaling:
            scale_factor = random.uniform(*self.scale_range)
            scale_factors = [scale_factor] * 3
        else:
            scale_factors = [random.uniform(*self.scale_range) for _ in range(3)]
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"] and len(tensor.shape) >= 3:
                result[key] = self._scale_tensor(tensor.to(self.device), scale_factors)
            else:
                result[key] = tensor
        
        return result
    
    def _scale_tensor(self, tensor: torch.Tensor, scale_factors: List[float]) -> torch.Tensor:
        """Scale tensor using interpolation."""
        original_shape = tensor.shape
        
        # Handle different tensor shapes
        if len(tensor.shape) == 3:  # (D, H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            spatial_dims = tensor.shape[2:]
        elif len(tensor.shape) == 4:  # (D, H, W, C) or (B, D, H, W)
            if tensor.shape[-1] <= 10:  # Assume last dim is channels
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
                spatial_dims = tensor.shape[2:]
            else:
                tensor = tensor.unsqueeze(1)  # (B, 1, D, H, W)
                spatial_dims = tensor.shape[2:]
        elif len(tensor.shape) == 5:  # (B, D, H, W, C)
            tensor = tensor.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
            spatial_dims = tensor.shape[2:]
        else:
            return tensor  # Unsupported shape
        
        # Calculate new size
        new_size = tuple(int(dim * scale) for dim, scale in zip(spatial_dims, scale_factors))
        
        # Apply scaling
        scaled = F.interpolate(
            tensor,
            size=new_size,
            mode='trilinear',
            align_corners=False
        )
        
        # Crop or pad to original size
        scaled = self._crop_or_pad_to_size(scaled, tensor.shape)
        
        # Restore original shape
        if len(original_shape) == 3:
            scaled = scaled.squeeze(0).squeeze(0)
        elif len(original_shape) == 4:
            if original_shape[-1] <= 10:  # Channels last
                scaled = scaled.squeeze(0).permute(1, 2, 3, 0)
            else:
                scaled = scaled.squeeze(1)
        elif len(original_shape) == 5:
            scaled = scaled.permute(0, 2, 3, 4, 1)
        
        return scaled
    
    def _crop_or_pad_to_size(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Crop or pad tensor to target shape."""
        current_shape = tensor.shape
        
        # Calculate padding/cropping for spatial dimensions
        pad_list = []
        for i in range(2, len(current_shape)):  # Skip batch and channel dims
            current_size = current_shape[i]
            target_size = target_shape[i]
            
            if current_size < target_size:
                # Need padding
                pad_total = target_size - current_size
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                pad_list.extend([pad_left, pad_right])
            elif current_size > target_size:
                # Need cropping - will handle after padding
                pad_list.extend([0, 0])
            else:
                pad_list.extend([0, 0])
        
        # Apply padding
        if any(p > 0 for p in pad_list):
            tensor = F.pad(tensor, pad_list[::-1])  # F.pad expects reversed order
        
        # Apply cropping if needed
        slices = [slice(None)] * len(tensor.shape)
        for i in range(2, len(current_shape)):
            current_size = tensor.shape[i]
            target_size = target_shape[i]
            
            if current_size > target_size:
                crop_total = current_size - target_size
                crop_left = crop_total // 2
                crop_right = current_size - crop_left
                slices[i] = slice(crop_left, crop_right)
        
        return tensor[tuple(slices)]


class NoiseAugmentation(BaseAugmentation):
    """
    Noise injection augmentation for sculpture data.
    """
    
    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_std: float = 0.01,
        noise_range: Optional[Tuple[float, float]] = None,
        probability: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize noise augmentation.
        
        Args:
            noise_type: Type of noise ("gaussian", "uniform", "salt_pepper")
            noise_std: Standard deviation for Gaussian noise
            noise_range: Range for uniform noise
            probability: Probability of applying augmentation
            device: Device for tensor operations
        """
        super().__init__(probability, device)
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_range = noise_range or (-0.05, 0.05)
    
    def apply(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply noise augmentation."""
        result = {}
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"]:
                result[key] = self._add_noise(tensor.to(self.device))
            else:
                result[key] = tensor
        
        return result
    
    def _add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add noise to tensor."""
        if self.noise_type == "gaussian":
            noise = torch.randn_like(tensor) * self.noise_std
            return tensor + noise
        
        elif self.noise_type == "uniform":
            noise = torch.rand_like(tensor) * (self.noise_range[1] - self.noise_range[0]) + self.noise_range[0]
            return tensor + noise
        
        elif self.noise_type == "salt_pepper":
            # Salt and pepper noise
            noise_mask = torch.rand_like(tensor) < self.noise_std
            salt_mask = torch.rand_like(tensor) < 0.5
            
            noisy_tensor = tensor.clone()
            noisy_tensor[noise_mask & salt_mask] = 1.0  # Salt
            noisy_tensor[noise_mask & ~salt_mask] = 0.0  # Pepper
            
            return noisy_tensor
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")


class FlipAugmentation(BaseAugmentation):
    """
    Random flipping augmentation for sculpture data.
    """
    
    def __init__(
        self,
        axes: List[int] = [0, 1, 2],
        probability: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize flip augmentation.
        
        Args:
            axes: List of axes to potentially flip
            probability: Probability of applying augmentation
            device: Device for tensor operations
        """
        super().__init__(probability, device)
        self.axes = axes
    
    def apply(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply flip augmentation."""
        result = {}
        
        # Choose random axis to flip
        flip_axis = random.choice(self.axes)
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"] and len(tensor.shape) >= 3:
                result[key] = self._flip_tensor(tensor.to(self.device), flip_axis)
            else:
                result[key] = tensor
        
        return result
    
    def _flip_tensor(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Flip tensor along specified axis."""
        # Adjust axis for tensor shape
        if len(tensor.shape) == 4 and tensor.shape[-1] <= 10:  # (D, H, W, C)
            # Don't flip channel dimension
            if axis >= 3:
                return tensor
        elif len(tensor.shape) == 5:  # (B, D, H, W, C)
            # Adjust axis to skip batch dimension
            axis += 1
            if axis >= 4:  # Don't flip channel dimension
                return tensor
        
        return torch.flip(tensor, dims=[axis])


class ElasticDeformationAugmentation(BaseAugmentation):
    """
    Elastic deformation augmentation for sculpture data.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 0.1,
        probability: float = 0.3,
        device: str = "cuda"
    ):
        """
        Initialize elastic deformation augmentation.
        
        Args:
            alpha: Scaling factor for deformation
            sigma: Standard deviation for Gaussian filter
            probability: Probability of applying augmentation
            device: Device for tensor operations
        """
        super().__init__(probability, device)
        self.alpha = alpha
        self.sigma = sigma
    
    def apply(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply elastic deformation augmentation."""
        result = {}
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"] and len(tensor.shape) >= 3:
                result[key] = self._elastic_deform(tensor.to(self.device))
            else:
                result[key] = tensor
        
        return result
    
    def _elastic_deform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation to tensor."""
        # Generate random displacement fields
        shape = tensor.shape
        
        # Handle different tensor shapes
        if len(shape) == 3:  # (D, H, W)
            spatial_shape = shape
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        elif len(shape) == 4:  # (D, H, W, C) or (B, D, H, W)
            if shape[-1] <= 10:  # Assume channels last
                spatial_shape = shape[:-1]
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
            else:
                spatial_shape = shape[1:]
                tensor = tensor.unsqueeze(1)  # (B, 1, D, H, W)
        elif len(shape) == 5:  # (B, D, H, W, C)
            spatial_shape = shape[1:-1]
            tensor = tensor.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        else:
            return tensor
        
        # Generate displacement fields
        dx = torch.randn(spatial_shape, device=self.device) * self.alpha
        dy = torch.randn(spatial_shape, device=self.device) * self.alpha
        dz = torch.randn(spatial_shape, device=self.device) * self.alpha
        
        # Apply Gaussian smoothing (simplified)
        kernel_size = max(3, int(self.sigma * 6) | 1)  # Ensure odd kernel size
        if kernel_size > 1:
            # Simple box filter approximation
            dx = F.avg_pool3d(dx.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2).squeeze()
            dy = F.avg_pool3d(dy.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2).squeeze()
            dz = F.avg_pool3d(dz.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2).squeeze()
        
        # Create coordinate grids
        d, h, w = spatial_shape
        z_coords = torch.linspace(-1, 1, d, device=self.device)
        y_coords = torch.linspace(-1, 1, h, device=self.device)
        x_coords = torch.linspace(-1, 1, w, device=self.device)
        
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Apply displacements
        zz_displaced = zz + dz * 2.0 / d  # Normalize displacement
        yy_displaced = yy + dy * 2.0 / h
        xx_displaced = xx + dx * 2.0 / w
        
        # Stack to create grid
        grid = torch.stack([xx_displaced, yy_displaced, zz_displaced], dim=-1)
        grid = grid.unsqueeze(0).expand(tensor.shape[0], -1, -1, -1, -1)
        
        # Apply deformation
        deformed = F.grid_sample(
            tensor,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Restore original shape
        if len(shape) == 3:
            deformed = deformed.squeeze(0).squeeze(0)
        elif len(shape) == 4:
            if shape[-1] <= 10:  # Channels last
                deformed = deformed.squeeze(0).permute(1, 2, 3, 0)
            else:
                deformed = deformed.squeeze(1)
        elif len(shape) == 5:
            deformed = deformed.permute(0, 2, 3, 4, 1)
        
        return deformed


class CompositeAugmentation:
    """
    Composite augmentation that applies multiple augmentations in sequence.
    """
    
    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        apply_probability: float = 1.0,
        device: str = "cuda"
    ):
        """
        Initialize composite augmentation.
        
        Args:
            augmentations: List of augmentations to apply
            apply_probability: Probability of applying the entire pipeline
            device: Device for tensor operations
        """
        self.augmentations = augmentations
        self.apply_probability = apply_probability
        self.device = device
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply composite augmentation."""
        if random.random() > self.apply_probability:
            return data
        
        result = data
        for augmentation in self.augmentations:
            result = augmentation(result)
        
        return result
    
    def to(self, device: str):
        """Move all augmentations to device."""
        self.device = device
        for augmentation in self.augmentations:
            augmentation.to(device)
        return self


# Convenience functions for creating common augmentation pipelines
def create_standard_augmentation_pipeline(
    device: str = "cuda",
    strong_augmentation: bool = False
) -> CompositeAugmentation:
    """Create a standard augmentation pipeline."""
    if strong_augmentation:
        augmentations = [
            RotationAugmentation(rotation_range=(-45, 45), probability=0.7, device=device),
            ScalingAugmentation(scale_range=(0.7, 1.3), probability=0.6, device=device),
            FlipAugmentation(probability=0.5, device=device),
            NoiseAugmentation(noise_std=0.02, probability=0.4, device=device),
            ElasticDeformationAugmentation(alpha=2.0, sigma=0.15, probability=0.3, device=device)
        ]
    else:
        augmentations = [
            RotationAugmentation(rotation_range=(-15, 15), probability=0.5, device=device),
            ScalingAugmentation(scale_range=(0.9, 1.1), probability=0.4, device=device),
            FlipAugmentation(probability=0.3, device=device),
            NoiseAugmentation(noise_std=0.01, probability=0.3, device=device)
        ]
    
    return CompositeAugmentation(augmentations, device=device)


def create_training_augmentation_pipeline(device: str = "cuda") -> CompositeAugmentation:
    """Create an augmentation pipeline optimized for training."""
    augmentations = [
        RotationAugmentation(rotation_range=(-20, 20), probability=0.6, device=device),
        ScalingAugmentation(scale_range=(0.85, 1.15), probability=0.5, device=device),
        FlipAugmentation(probability=0.4, device=device),
        NoiseAugmentation(noise_type="gaussian", noise_std=0.015, probability=0.3, device=device)
    ]
    
    return CompositeAugmentation(augmentations, apply_probability=0.8, device=device)


def create_geometric_augmentation_pipeline(device: str = "cuda") -> CompositeAugmentation:
    """Create a pipeline focused on geometric augmentations."""
    augmentations = [
        RotationAugmentation(rotation_range=(-30, 30), probability=0.8, device=device),
        ScalingAugmentation(scale_range=(0.8, 1.2), uniform_scaling=False, probability=0.6, device=device),
        FlipAugmentation(probability=0.5, device=device),
        ElasticDeformationAugmentation(alpha=1.5, sigma=0.1, probability=0.4, device=device)
    ]
    
    return CompositeAugmentation(augmentations, device=device)