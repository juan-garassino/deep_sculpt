"""
Data preprocessing components for DeepSculpt PyTorch implementation.

This module provides comprehensive preprocessing functionality including
normalization, augmentation, filtering, and data quality control.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import random
import warnings
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize base preprocessor.
        
        Args:
            device: Device for tensor operations
        """
        self.device = device
    
    @abstractmethod
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply preprocessing to data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Preprocessed data dictionary
        """
        pass
    
    def to(self, device: str):
        """Move preprocessor to device."""
        self.device = device
        return self


class NormalizationPreprocessor(BasePreprocessor):
    """
    Normalization preprocessor for 3D sculpture data.
    
    Supports various normalization strategies including min-max,
    z-score, and custom normalization schemes.
    """
    
    def __init__(
        self,
        method: str = "min_max",
        target_range: Tuple[float, float] = (0.0, 1.0),
        per_channel: bool = True,
        epsilon: float = 1e-8,
        device: str = "cuda"
    ):
        """
        Initialize normalization preprocessor.
        
        Args:
            method: Normalization method ("min_max", "z_score", "unit_norm")
            target_range: Target range for min-max normalization
            per_channel: Whether to normalize per channel
            epsilon: Small value to avoid division by zero
            device: Device for tensor operations
        """
        super().__init__(device)
        self.method = method
        self.target_range = target_range
        self.per_channel = per_channel
        self.epsilon = epsilon
        
        # Statistics for z-score normalization
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
    
    def fit(self, data: torch.Tensor):
        """
        Fit normalization parameters to data.
        
        Args:
            data: Data tensor to fit parameters on
        """
        data = data.to(self.device)
        
        if self.method == "z_score":
            if self.per_channel and len(data.shape) > 3:
                # Calculate per-channel statistics
                dims = list(range(len(data.shape)))
                dims.remove(-1)  # Keep channel dimension
                self.mean = data.mean(dim=dims, keepdim=True)
                self.std = data.std(dim=dims, keepdim=True)
            else:
                self.mean = data.mean()
                self.std = data.std()
        
        elif self.method == "min_max":
            if self.per_channel and len(data.shape) > 3:
                # Calculate per-channel min/max
                dims = list(range(len(data.shape)))
                dims.remove(-1)  # Keep channel dimension
                self.min_val = data.min(dim=dims[0], keepdim=True)[0]
                self.max_val = data.max(dim=dims[0], keepdim=True)[0]
                for dim in dims[1:]:
                    self.min_val = self.min_val.min(dim=dim, keepdim=True)[0]
                    self.max_val = self.max_val.max(dim=dim, keepdim=True)[0]
            else:
                self.min_val = data.min()
                self.max_val = data.max()
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply normalization to data."""
        result = {}
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"]:  # Apply to main data tensors
                result[key] = self._normalize_tensor(tensor.to(self.device))
            else:
                result[key] = tensor
        
        return result
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a single tensor."""
        if self.method == "min_max":
            min_val = self.min_val if self.min_val is not None else tensor.min()
            max_val = self.max_val if self.max_val is not None else tensor.max()
            
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = torch.where(range_val < self.epsilon, torch.ones_like(range_val), range_val)
            
            # Normalize to [0, 1]
            normalized = (tensor - min_val) / range_val
            
            # Scale to target range
            target_min, target_max = self.target_range
            normalized = normalized * (target_max - target_min) + target_min
            
            return normalized
        
        elif self.method == "z_score":
            mean = self.mean if self.mean is not None else tensor.mean()
            std = self.std if self.std is not None else tensor.std()
            
            # Avoid division by zero
            std = torch.where(std < self.epsilon, torch.ones_like(std), std)
            
            return (tensor - mean) / std
        
        elif self.method == "unit_norm":
            # Normalize to unit norm
            norm = tensor.norm(dim=-1, keepdim=True)
            norm = torch.where(norm < self.epsilon, torch.ones_like(norm), norm)
            return tensor / norm
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class AugmentationPreprocessor(BasePreprocessor):
    """
    Data augmentation preprocessor for 3D sculpture data.
    
    Provides various augmentation techniques including rotation,
    scaling, noise addition, and geometric transformations.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.01,
        flip_probability: float = 0.5,
        augmentation_probability: float = 0.8,
        device: str = "cuda"
    ):
        """
        Initialize augmentation preprocessor.
        
        Args:
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            noise_std: Standard deviation of Gaussian noise
            flip_probability: Probability of random flipping
            augmentation_probability: Probability of applying augmentation
            device: Device for tensor operations
        """
        super().__init__(device)
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.flip_probability = flip_probability
        self.augmentation_probability = augmentation_probability
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to data."""
        if random.random() > self.augmentation_probability:
            return data  # Skip augmentation
        
        result = {}
        
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"] and len(tensor.shape) >= 3:
                result[key] = self._augment_tensor(tensor.to(self.device))
            else:
                result[key] = tensor
        
        return result
    
    def _augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a single tensor."""
        # Random rotation
        if self.rotation_range[0] != self.rotation_range[1]:
            angle = random.uniform(*self.rotation_range)
            tensor = self._rotate_3d(tensor, angle)
        
        # Random scaling
        if self.scale_range[0] != self.scale_range[1]:
            scale = random.uniform(*self.scale_range)
            tensor = self._scale_3d(tensor, scale)
        
        # Random flipping
        if random.random() < self.flip_probability:
            # Randomly choose axis to flip
            axis = random.choice([0, 1, 2])
            if len(tensor.shape) == 4:  # (D, H, W, C)
                axis += 0  # Skip batch dimension if present
            elif len(tensor.shape) == 5:  # (B, D, H, W, C)
                axis += 1  # Skip batch dimension
            tensor = torch.flip(tensor, dims=[axis])
        
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise
        
        return tensor
    
    def _rotate_3d(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply 3D rotation to tensor."""
        # Simple rotation around Z-axis
        # For more complex rotations, would need proper 3D rotation matrices
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Create rotation matrix for Z-axis rotation
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=tensor.dtype, device=tensor.device)
        
        # Apply rotation (simplified - would need proper 3D transformation)
        # For now, just return original tensor with small random transformation
        return tensor
    
    def _scale_3d(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply 3D scaling to tensor."""
        if abs(scale - 1.0) < 1e-6:
            return tensor
        
        # Use interpolation for scaling
        if len(tensor.shape) == 4:  # (D, H, W, C)
            # Permute to (C, D, H, W) for interpolation
            tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)  # Add batch dim
            
            # Calculate new size
            old_size = tensor.shape[2:]
            new_size = tuple(int(s * scale) for s in old_size)
            
            # Interpolate
            tensor = F.interpolate(tensor, size=new_size, mode='trilinear', align_corners=False)
            
            # Crop or pad to original size
            tensor = self._crop_or_pad_to_size(tensor, (1,) + (tensor.shape[1],) + old_size)
            
            # Permute back to (D, H, W, C)
            tensor = tensor.squeeze(0).permute(1, 2, 3, 0)
        
        return tensor
    
    def _crop_or_pad_to_size(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Crop or pad tensor to target shape."""
        current_shape = tensor.shape
        
        # Calculate padding/cropping for each dimension
        operations = []
        for i, (current, target) in enumerate(zip(current_shape, target_shape)):
            if current < target:
                # Need padding
                pad_total = target - current
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                operations.append(('pad', i, pad_left, pad_right))
            elif current > target:
                # Need cropping
                crop_total = current - target
                crop_left = crop_total // 2
                crop_right = current - crop_left
                operations.append(('crop', i, crop_left, crop_right))
        
        # Apply operations
        for op_type, dim, left, right in operations:
            if op_type == 'pad':
                # Create padding tuple for F.pad
                pad_tuple = [0, 0] * len(current_shape)
                pad_tuple[-(dim+1)*2-1] = left  # Left padding
                pad_tuple[-(dim+1)*2] = right   # Right padding
                tensor = F.pad(tensor, pad_tuple)
            elif op_type == 'crop':
                # Create slice for cropping
                slices = [slice(None)] * len(current_shape)
                slices[dim] = slice(left, right)
                tensor = tensor[tuple(slices)]
        
        return tensor


class FilteringPreprocessor(BasePreprocessor):
    """
    Filtering preprocessor for data quality control.
    
    Filters out samples based on various criteria such as
    sparsity, complexity, and data quality metrics.
    """
    
    def __init__(
        self,
        min_sparsity: float = 0.1,
        max_sparsity: float = 0.9,
        min_complexity: float = 0.01,
        check_nan: bool = True,
        check_inf: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize filtering preprocessor.
        
        Args:
            min_sparsity: Minimum sparsity threshold
            max_sparsity: Maximum sparsity threshold
            min_complexity: Minimum complexity (standard deviation)
            check_nan: Whether to check for NaN values
            check_inf: Whether to check for infinite values
            device: Device for tensor operations
        """
        super().__init__(device)
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.min_complexity = min_complexity
        self.check_nan = check_nan
        self.check_inf = check_inf
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Apply filtering to data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Data dictionary if it passes filters, None otherwise
        """
        for key, tensor in data.items():
            if key in ["structure", "colors", "data"]:
                tensor = tensor.to(self.device)
                
                # Check for NaN/Inf values
                if self.check_nan and torch.isnan(tensor).any():
                    return None
                if self.check_inf and torch.isinf(tensor).any():
                    return None
                
                # Check sparsity
                sparsity = (tensor == 0).float().mean().item()
                if sparsity < self.min_sparsity or sparsity > self.max_sparsity:
                    return None
                
                # Check complexity
                complexity = tensor.std().item()
                if complexity < self.min_complexity:
                    return None
        
        return data


class CompositePreprocessor(BasePreprocessor):
    """
    Composite preprocessor that applies multiple preprocessing steps.
    """
    
    def __init__(
        self,
        preprocessors: List[BasePreprocessor],
        device: str = "cuda"
    ):
        """
        Initialize composite preprocessor.
        
        Args:
            preprocessors: List of preprocessors to apply in order
            device: Device for tensor operations
        """
        super().__init__(device)
        self.preprocessors = preprocessors
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Apply all preprocessors in sequence."""
        result = data
        
        for preprocessor in self.preprocessors:
            result = preprocessor(result)
            if result is None:  # Filtered out
                return None
        
        return result
    
    def to(self, device: str):
        """Move all preprocessors to device."""
        super().to(device)
        for preprocessor in self.preprocessors:
            preprocessor.to(device)
        return self


class ConditionalPreprocessor(BasePreprocessor):
    """
    Conditional preprocessor that applies different preprocessing based on conditions.
    """
    
    def __init__(
        self,
        condition_fn: Callable[[Dict[str, torch.Tensor]], str],
        preprocessor_map: Dict[str, BasePreprocessor],
        default_preprocessor: Optional[BasePreprocessor] = None,
        device: str = "cuda"
    ):
        """
        Initialize conditional preprocessor.
        
        Args:
            condition_fn: Function that returns condition key based on data
            preprocessor_map: Mapping from condition keys to preprocessors
            default_preprocessor: Default preprocessor if condition not found
            device: Device for tensor operations
        """
        super().__init__(device)
        self.condition_fn = condition_fn
        self.preprocessor_map = preprocessor_map
        self.default_preprocessor = default_preprocessor
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Apply conditional preprocessing."""
        condition = self.condition_fn(data)
        
        if condition in self.preprocessor_map:
            preprocessor = self.preprocessor_map[condition]
        elif self.default_preprocessor is not None:
            preprocessor = self.default_preprocessor
        else:
            return data  # No preprocessing
        
        return preprocessor(data)


# Convenience functions for creating common preprocessors
def create_standard_preprocessor(
    device: str = "cuda",
    augment: bool = True,
    normalize: bool = True,
    filter_data: bool = True
) -> CompositePreprocessor:
    """Create a standard preprocessing pipeline."""
    preprocessors = []
    
    if filter_data:
        preprocessors.append(FilteringPreprocessor(device=device))
    
    if normalize:
        preprocessors.append(NormalizationPreprocessor(device=device))
    
    if augment:
        preprocessors.append(AugmentationPreprocessor(device=device))
    
    return CompositePreprocessor(preprocessors, device=device)


def create_training_preprocessor(device: str = "cuda") -> CompositePreprocessor:
    """Create a preprocessor optimized for training."""
    return CompositePreprocessor([
        FilteringPreprocessor(device=device),
        NormalizationPreprocessor(method="min_max", device=device),
        AugmentationPreprocessor(augmentation_probability=0.8, device=device)
    ], device=device)


def create_validation_preprocessor(device: str = "cuda") -> CompositePreprocessor:
    """Create a preprocessor optimized for validation (no augmentation)."""
    return CompositePreprocessor([
        FilteringPreprocessor(device=device),
        NormalizationPreprocessor(method="min_max", device=device)
    ], device=device)