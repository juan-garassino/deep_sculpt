"""
PyTorch Utility Functions for DeepSculpt Geometry Operations
This module provides PyTorch tensor-based utility functions for 3D shape manipulation,
coordinate handling, random geometry generation, and debugging utilities.

Key features:
- Random dimension generation: Functions for creating size parameters within constraints
- Coordinate operations: Functions for position selection and validation using PyTorch tensors
- Shape insertion: Utilities for adding shapes to 3D tensors
- Validation: Functions for checking geometric constraints and boundaries
- Debug utilities: Tools for inspecting and reporting on 3D tensor structures
- Memory optimization: Automatic sparse/dense tensor conversion and memory monitoring
- GPU memory management: Functions for efficient GPU memory usage

Dependencies:
- torch: For tensor operations and GPU acceleration
- logger.py: For detailed operation logging
- numpy: For compatibility with existing code

Used by:
- pytorch_shapes.py: For coordinate operations and validation during shape creation
- pytorch_visualization.py: For data transformation before visualization
- pytorch_sculptor.py: For geometry validation and manipulation
- pytorch_collector.py: For tensor operations during dataset generation
- pytorch_curator.py: For data inspection and validation
"""

import torch
import numpy as np
import random
import psutil
import gc
from typing import Tuple, List, Optional, Any, Dict, Union
from .logger import log_info, log_error, log_warning, begin_section, end_section


class PyTorchUtils:
    """
    PyTorch-based utility functions for 3D tensor operations and memory management.
    """
    
    @staticmethod
    def return_axis(
        void: torch.Tensor, 
        color_void: torch.Tensor,
        device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Selects a random plane from a 3D PyTorch tensor along a random axis.

        Args:
            void: The 3D PyTorch tensor to select a plane from.
            color_void: The 3D PyTorch tensor that holds the color information.
            device: Target device for tensors (if None, uses void's device)

        Returns:
            A tuple containing:
                - working_plane: The randomly selected plane.
                - color_parameters: The color information of the selected plane.
                - section: The index of the selected plane.
        """
        if device is None:
            device = void.device
            
        # Ensure tensors are on the correct device
        void = void.to(device)
        color_void = color_void.to(device)
        
        section = torch.randint(low=0, high=void.shape[0], size=(1,), device=device).item()
        axis_selection = torch.randint(low=0, high=3, size=(1,), device=device).item()

        log_info(f"Selected axis {axis_selection}, section {section}", is_last=False)

        if axis_selection == 0:
            working_plane = void[section, :, :]
            color_parameters = color_void[section, :, :]
        elif axis_selection == 1:
            working_plane = void[:, section, :]
            color_parameters = color_void[:, section, :]
        elif axis_selection == 2:
            working_plane = void[:, :, section]
            color_parameters = color_void[:, :, section]
        else:
            log_error("Axis selection value out of range.")
            raise ValueError("Axis selection value out of range.")

        return working_plane, color_parameters, section

    @staticmethod
    def generate_random_size(
        min_ratio: float, 
        max_ratio: float, 
        base_size: int, 
        step: int = 1,
        device: str = "cpu"
    ) -> int:
        """
        Generate a random size based on given ratios and base size using PyTorch.

        Args:
            min_ratio: Minimum size ratio relative to base_size
            max_ratio: Maximum size ratio relative to base_size
            base_size: Reference size (usually the smallest dimension of the void)
            step: Step size for the random range
            device: Device for tensor operations

        Returns:
            Integer representing the random size
        """
        min_size = max(int(min_ratio * base_size), 2)  # Ensure minimum size of 2
        max_size = max(int(max_ratio * base_size), min_size + 1)  # Ensure max > min

        if step > 1:
            # Adjust to be multiples of step
            min_size = (min_size // step) * step
            max_size = (max_size // step) * step
            if min_size == max_size:
                return min_size

        # Use PyTorch for random generation
        size_range = torch.arange(min_size, max_size, step, device=device)
        if len(size_range) == 0:
            return min_size
        
        random_idx = torch.randint(0, len(size_range), (1,), device=device)
        return size_range[random_idx].item()

    @staticmethod
    def select_random_position(
        max_pos: int, 
        size: int, 
        device: str = "cpu"
    ) -> int:
        """
        Select a random position to insert a shape within bounds using PyTorch.

        Args:
            max_pos: Maximum position value (usually dimension size)
            size: Size of the shape to be inserted
            device: Device for tensor operations

        Returns:
            Integer representing the random position
        """
        max_valid_pos = max(0, max_pos - size)
        if max_valid_pos == 0:
            return 0
        
        return torch.randint(0, max_valid_pos + 1, (1,), device=device).item()

    @staticmethod
    def insert_shape(
        void: torch.Tensor, 
        shape_indices: tuple, 
        values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Insert a shape into the void tensor at the given indices.

        Args:
            void: 3D PyTorch tensor representing the space
            shape_indices: Tuple of slices or indices where to insert the shape
            values: Values to insert, if None uses 1s

        Returns:
            Updated void tensor with the shape inserted
        """
        if values is None:
            void[shape_indices] = 1
        else:
            # Ensure values are on the same device as void
            if isinstance(values, torch.Tensor):
                values = values.to(void.device)
            void[shape_indices] = values
        return void

    @staticmethod
    def assign_color(
        color_void: torch.Tensor, 
        shape_indices: tuple, 
        color: Any
    ) -> torch.Tensor:
        """
        Assign color to the shape in the color void tensor.

        Args:
            color_void: 3D PyTorch tensor representing colors
            shape_indices: Tuple of slices or indices where the shape is
            color: Color to assign to the shape

        Returns:
            Updated color_void tensor with colors assigned to the shape
        """
        if isinstance(color, torch.Tensor):
            color = color.to(color_void.device)
        color_void[shape_indices] = color
        return color_void

    @staticmethod
    def validate_dimensions(
        shape_size: List[int], 
        void_shape: Tuple[int, ...]
    ) -> bool:
        """
        Validate that the shape fits within the void dimensions.

        Args:
            shape_size: Dimensions of the shape
            void_shape: Dimensions of the void

        Returns:
            Boolean indicating if the shape fits in the void
        """
        return all(s <= v for s, v in zip(shape_size, void_shape))

    @staticmethod
    def validate_bounds(
        start_pos: List[int], 
        shape_size: List[int], 
        void_shape: Tuple[int, ...]
    ) -> bool:
        """
        Validate that the shape at the given position fits within the void bounds.

        Args:
            start_pos: Starting position coordinates
            shape_size: Dimensions of the shape
            void_shape: Dimensions of the void

        Returns:
            Boolean indicating if the shape at the position fits in the void
        """
        for i in range(len(start_pos)):
            if start_pos[i] < 0 or start_pos[i] + shape_size[i] > void_shape[i]:
                return False
        return True

    @staticmethod
    def select_random_color(colors: List[str]) -> str:
        """
        Select a random color from a list or return the color if it's a string.

        Args:
            colors: List of color strings or a single color string

        Returns:
            Selected color string
        """
        if isinstance(colors, list):
            return random.choice(colors)
        return colors

    @staticmethod
    def tensor_to_voxel_coordinates(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a 3D tensor to voxel coordinates for filled positions.

        Args:
            tensor: 3D PyTorch tensor

        Returns:
            Tensor of shape (N, 3) containing coordinates of filled voxels
        """
        # Find non-zero positions
        nonzero_indices = torch.nonzero(tensor > 0, as_tuple=False)
        return nonzero_indices

    @staticmethod
    def apply_3d_transformations(
        tensor: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply 3D transformations to a tensor.

        Args:
            tensor: Input 3D tensor
            rotation: 3x3 rotation matrix (optional)
            translation: 3D translation vector (optional)
            scale: Scaling factor (optional)

        Returns:
            Transformed tensor
        """
        device = tensor.device
        
        # Get coordinates of filled voxels
        coords = PyTorchUtils.tensor_to_voxel_coordinates(tensor)
        if len(coords) == 0:
            return tensor
        
        # Convert to float for transformations
        coords_float = coords.float()
        
        # Apply scaling
        if scale is not None:
            coords_float *= scale
        
        # Apply rotation
        if rotation is not None:
            rotation = rotation.to(device)
            coords_float = torch.matmul(coords_float, rotation.T)
        
        # Apply translation
        if translation is not None:
            translation = translation.to(device)
            coords_float += translation
        
        # Round back to integer coordinates
        coords_transformed = torch.round(coords_float).long()
        
        # Create new tensor with transformed coordinates
        # For simplicity, we'll create a tensor of the same size
        # In practice, you might want to adjust the size based on transformations
        result = torch.zeros_like(tensor)
        
        # Filter coordinates that are within bounds
        valid_mask = (
            (coords_transformed >= 0).all(dim=1) &
            (coords_transformed[:, 0] < tensor.shape[0]) &
            (coords_transformed[:, 1] < tensor.shape[1]) &
            (coords_transformed[:, 2] < tensor.shape[2])
        )
        
        valid_coords = coords_transformed[valid_mask]
        original_coords = coords[valid_mask]
        
        if len(valid_coords) > 0:
            # Copy values from original positions to new positions
            result[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = \
                tensor[original_coords[:, 0], original_coords[:, 1], original_coords[:, 2]]
        
        return result

    @staticmethod
    def validate_tensor_dtype(
        tensor: torch.Tensor, 
        expected_dtype: torch.dtype
    ) -> bool:
        """
        Validate that a tensor has the expected dtype.

        Args:
            tensor: PyTorch tensor to validate
            expected_dtype: Expected data type

        Returns:
            Boolean indicating if the tensor has the expected dtype
        """
        return tensor.dtype == expected_dtype

    @staticmethod
    def validate_tensor_device(
        tensor: torch.Tensor, 
        expected_device: str
    ) -> bool:
        """
        Validate that a tensor is on the expected device.

        Args:
            tensor: PyTorch tensor to validate
            expected_device: Expected device (e.g., 'cpu', 'cuda:0')

        Returns:
            Boolean indicating if the tensor is on the expected device
        """
        return str(tensor.device) == expected_device

    @staticmethod
    def ensure_tensor_device(
        tensor: torch.Tensor, 
        device: str
    ) -> torch.Tensor:
        """
        Ensure a tensor is on the specified device.

        Args:
            tensor: PyTorch tensor
            device: Target device

        Returns:
            Tensor on the specified device
        """
        return tensor.to(device)

    @staticmethod
    def efficient_tensor_slicing(
        tensor: torch.Tensor, 
        slice_indices: Tuple[slice, ...]
    ) -> torch.Tensor:
        """
        Perform efficient tensor slicing with bounds checking.

        Args:
            tensor: Input tensor
            slice_indices: Tuple of slice objects

        Returns:
            Sliced tensor
        """
        # Validate slice indices
        for i, slice_obj in enumerate(slice_indices):
            if i >= len(tensor.shape):
                raise IndexError(f"Slice index {i} out of range for tensor with {len(tensor.shape)} dimensions")
            
            if slice_obj.start is not None and slice_obj.start < 0:
                raise IndexError(f"Negative slice start {slice_obj.start} not supported")
            
            if slice_obj.stop is not None and slice_obj.stop > tensor.shape[i]:
                raise IndexError(f"Slice stop {slice_obj.stop} exceeds dimension size {tensor.shape[i]}")
        
        return tensor[slice_indices]

    @staticmethod
    def create_debug_info(
        tensor: torch.Tensor, 
        filled_only: bool = True
    ) -> Dict[str, Any]:
        """
        Create debug information about the tensor.

        Args:
            tensor: The 3D PyTorch tensor
            filled_only: If True, only count filled voxels

        Returns:
            Dictionary with debug information
        """
        info = {
            "shape": tuple(tensor.shape),
            "total_voxels": tensor.numel(),
            "device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "memory_usage_bytes": tensor.element_size() * tensor.numel(),
        }

        if filled_only:
            filled = tensor > 0
            info["filled_voxels"] = torch.sum(filled).item()
            info["fill_percentage"] = (info["filled_voxels"] / info["total_voxels"]) * 100
            info["sparsity"] = 1.0 - (info["filled_voxels"] / info["total_voxels"])

        return info

    @staticmethod
    def print_debug_info(info: Dict[str, Any]):
        """
        Print debug information in a structured format.

        Args:
            info: Dictionary with debug information
        """
        begin_section("PyTorch Tensor Debug Information")

        for key, value in info.items():
            if key == "fill_percentage":
                log_info(f"{key}: {value:.2f}%")
            elif key == "sparsity":
                log_info(f"{key}: {value:.4f}")
            elif key == "memory_usage_bytes":
                # Convert to human-readable format
                if value > 1024**3:
                    log_info(f"{key}: {value / (1024**3):.2f} GB")
                elif value > 1024**2:
                    log_info(f"{key}: {value / (1024**2):.2f} MB")
                elif value > 1024:
                    log_info(f"{key}: {value / 1024:.2f} KB")
                else:
                    log_info(f"{key}: {value} bytes")
            else:
                log_info(f"{key}: {value}")

        end_section()


class MemoryOptimizer:
    """
    Memory optimization utilities for PyTorch tensors including sparse/dense conversion,
    memory monitoring, and compression utilities.
    """
    
    @staticmethod
    def detect_sparsity(tensor: torch.Tensor) -> float:
        """
        Detect the sparsity level of a tensor.

        Args:
            tensor: Input PyTorch tensor

        Returns:
            Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
        """
        total_elements = tensor.numel()
        if total_elements == 0:
            return 0.0
        
        zero_elements = torch.sum(tensor == 0).item()
        return zero_elements / total_elements

    @staticmethod
    def should_use_sparse(
        tensor: torch.Tensor, 
        sparsity_threshold: float = 0.5,
        memory_threshold: float = 0.8
    ) -> bool:
        """
        Determine if a tensor should use sparse representation.

        Args:
            tensor: Input tensor
            sparsity_threshold: Minimum sparsity ratio to consider sparse conversion
            memory_threshold: Memory usage threshold (0.0-1.0) to trigger sparse conversion

        Returns:
            Boolean indicating if sparse representation should be used
        """
        sparsity = MemoryOptimizer.detect_sparsity(tensor)
        
        # Check sparsity threshold
        if sparsity < sparsity_threshold:
            return False
        
        # Check memory usage
        current_memory = MemoryOptimizer.get_tensor_memory_usage(tensor)
        available_memory = MemoryOptimizer.get_available_memory()
        
        if available_memory > 0:
            memory_usage_ratio = current_memory / available_memory
            if memory_usage_ratio > memory_threshold:
                return True
        
        return sparsity >= sparsity_threshold

    @staticmethod
    def to_sparse(
        tensor: torch.Tensor, 
        threshold: float = 0.01
    ) -> torch.sparse.FloatTensor:
        """
        Convert a dense tensor to sparse representation.

        Args:
            tensor: Dense PyTorch tensor
            threshold: Values below this threshold are considered zero

        Returns:
            Sparse tensor representation
        """
        # Apply threshold to create true zeros
        tensor_thresholded = torch.where(
            torch.abs(tensor) < threshold, 
            torch.zeros_like(tensor), 
            tensor
        )
        
        # Convert to sparse COO format
        sparse_tensor = tensor_thresholded.to_sparse()
        
        log_info(f"Converted tensor to sparse format. "
                f"Original size: {tensor.numel()}, "
                f"Sparse nnz: {sparse_tensor._nnz()}")
        
        return sparse_tensor

    @staticmethod
    def to_dense(sparse_tensor: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Convert a sparse tensor to dense representation.

        Args:
            sparse_tensor: Sparse PyTorch tensor

        Returns:
            Dense tensor representation
        """
        dense_tensor = sparse_tensor.to_dense()
        
        log_info(f"Converted sparse tensor to dense format. "
                f"Sparse nnz: {sparse_tensor._nnz()}, "
                f"Dense size: {dense_tensor.numel()}")
        
        return dense_tensor

    @staticmethod
    def auto_convert_sparse_dense(
        tensor: torch.Tensor,
        sparsity_threshold: float = 0.5,
        force_sparse: bool = False,
        force_dense: bool = False
    ) -> Union[torch.Tensor, torch.sparse.FloatTensor]:
        """
        Automatically convert between sparse and dense representations based on sparsity.

        Args:
            tensor: Input tensor (dense or sparse)
            sparsity_threshold: Threshold for automatic conversion
            force_sparse: Force conversion to sparse
            force_dense: Force conversion to dense

        Returns:
            Optimally represented tensor
        """
        if force_sparse and force_dense:
            raise ValueError("Cannot force both sparse and dense conversion")
        
        is_sparse = tensor.is_sparse
        
        if force_sparse and not is_sparse:
            return MemoryOptimizer.to_sparse(tensor)
        
        if force_dense and is_sparse:
            return MemoryOptimizer.to_dense(tensor)
        
        if is_sparse:
            # Check if we should convert to dense
            sparsity = 1.0 - (tensor._nnz() / tensor.numel())
            if sparsity < sparsity_threshold:
                log_info(f"Converting sparse tensor to dense (sparsity: {sparsity:.3f})")
                return MemoryOptimizer.to_dense(tensor)
        else:
            # Check if we should convert to sparse
            if MemoryOptimizer.should_use_sparse(tensor, sparsity_threshold):
                log_info(f"Converting dense tensor to sparse")
                return MemoryOptimizer.to_sparse(tensor)
        
        return tensor

    @staticmethod
    def get_tensor_memory_usage(tensor: torch.Tensor) -> int:
        """
        Get memory usage of a tensor in bytes.

        Args:
            tensor: PyTorch tensor

        Returns:
            Memory usage in bytes
        """
        if tensor.is_sparse:
            # For sparse tensors, calculate based on stored values and indices
            indices_memory = tensor._indices().element_size() * tensor._indices().numel()
            values_memory = tensor._values().element_size() * tensor._values().numel()
            return indices_memory + values_memory
        else:
            return tensor.element_size() * tensor.numel()

    @staticmethod
    def get_available_memory(device: str = "cuda") -> int:
        """
        Get available memory on the specified device.

        Args:
            device: Device to check ('cuda' or 'cpu')

        Returns:
            Available memory in bytes
        """
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        else:
            # For CPU, use system memory
            return psutil.virtual_memory().available

    @staticmethod
    def get_memory_stats(device: str = "cuda") -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Args:
            device: Device to check

        Returns:
            Dictionary with memory statistics
        """
        stats = {}
        
        if device.startswith("cuda") and torch.cuda.is_available():
            stats.update({
                "device": device,
                "total_memory": torch.cuda.get_device_properties(device).total_memory,
                "allocated_memory": torch.cuda.memory_allocated(device),
                "cached_memory": torch.cuda.memory_reserved(device),
                "available_memory": torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device),
            })
        else:
            vm = psutil.virtual_memory()
            stats.update({
                "device": "cpu",
                "total_memory": vm.total,
                "allocated_memory": vm.used,
                "available_memory": vm.available,
                "memory_percent": vm.percent,
            })
        
        return stats

    @staticmethod
    def optimize_memory_usage():
        """
        Perform memory optimization by clearing cache and running garbage collection.
        """
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        log_info("Memory optimization completed")

    @staticmethod
    def monitor_memory_usage(
        operation: callable,
        *args,
        device: str = "cuda",
        **kwargs
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Monitor memory usage during an operation.

        Args:
            operation: Function to monitor
            *args: Arguments for the operation
            device: Device to monitor
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, memory_stats)
        """
        # Get initial memory stats
        initial_stats = MemoryOptimizer.get_memory_stats(device)
        
        # Run the operation
        result = operation(*args, **kwargs)
        
        # Get final memory stats
        final_stats = MemoryOptimizer.get_memory_stats(device)
        
        # Calculate memory difference
        memory_diff = {
            "memory_increase": final_stats["allocated_memory"] - initial_stats["allocated_memory"],
            "peak_memory": final_stats["allocated_memory"],
            "initial_memory": initial_stats["allocated_memory"],
        }
        
        return result, memory_diff

    @staticmethod
    def suggest_optimization(tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Suggest memory optimizations for a tensor.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary with optimization suggestions
        """
        suggestions = {
            "current_memory": MemoryOptimizer.get_tensor_memory_usage(tensor),
            "sparsity": MemoryOptimizer.detect_sparsity(tensor),
            "is_sparse": tensor.is_sparse,
            "suggestions": []
        }
        
        sparsity = suggestions["sparsity"]
        
        if not tensor.is_sparse and sparsity > 0.5:
            potential_savings = suggestions["current_memory"] * sparsity
            suggestions["suggestions"].append({
                "type": "convert_to_sparse",
                "reason": f"High sparsity ({sparsity:.2%})",
                "potential_savings_bytes": potential_savings,
                "potential_savings_percent": sparsity * 100
            })
        
        if tensor.is_sparse and sparsity < 0.3:
            suggestions["suggestions"].append({
                "type": "convert_to_dense",
                "reason": f"Low sparsity ({sparsity:.2%})",
                "note": "Dense representation might be more efficient"
            })
        
        # Check dtype optimization
        if tensor.dtype == torch.float64:
            potential_savings = suggestions["current_memory"] * 0.5
            suggestions["suggestions"].append({
                "type": "reduce_precision",
                "reason": "Using float64, consider float32",
                "potential_savings_bytes": potential_savings,
                "potential_savings_percent": 50
            })
        
        return suggestions

    @staticmethod
    def compress_tensor(
        tensor: torch.Tensor,
        compression_method: str = "quantization",
        **kwargs
    ) -> torch.Tensor:
        """
        Compress a tensor using various methods.

        Args:
            tensor: Input tensor
            compression_method: Method to use ('quantization', 'pruning')
            **kwargs: Additional arguments for compression methods

        Returns:
            Compressed tensor
        """
        if compression_method == "quantization":
            # Simple quantization to int8
            scale = kwargs.get("scale", None)
            if scale is None:
                scale = tensor.abs().max() / 127.0
            
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            
            # Store scale for dequantization
            quantized.scale = scale
            
            log_info(f"Quantized tensor with scale {scale:.6f}")
            return quantized
            
        elif compression_method == "pruning":
            # Simple magnitude-based pruning
            threshold = kwargs.get("threshold", 0.01)
            pruned = torch.where(torch.abs(tensor) < threshold, torch.zeros_like(tensor), tensor)
            
            pruned_ratio = torch.sum(pruned == 0).float() / tensor.numel()
            log_info(f"Pruned {pruned_ratio:.2%} of tensor values")
            
            return pruned
        
        else:
            raise ValueError(f"Unknown compression method: {compression_method}")

    @staticmethod
    def create_memory_profile(
        tensors: List[torch.Tensor],
        names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a memory profile for a list of tensors.

        Args:
            tensors: List of tensors to profile
            names: Optional names for the tensors

        Returns:
            Dictionary with memory profile information
        """
        if names is None:
            names = [f"tensor_{i}" for i in range(len(tensors))]
        
        profile = {
            "total_tensors": len(tensors),
            "total_memory": 0,
            "tensor_details": []
        }
        
        for i, (tensor, name) in enumerate(zip(tensors, names)):
            memory_usage = MemoryOptimizer.get_tensor_memory_usage(tensor)
            sparsity = MemoryOptimizer.detect_sparsity(tensor)
            
            tensor_info = {
                "name": name,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "is_sparse": tensor.is_sparse,
                "memory_usage": memory_usage,
                "sparsity": sparsity,
                "suggestions": MemoryOptimizer.suggest_optimization(tensor)["suggestions"]
            }
            
            profile["tensor_details"].append(tensor_info)
            profile["total_memory"] += memory_usage
        
        return profile


class SparseTensorHandler:
    """
    Handles conversion between dense and sparse representations with automatic detection
    and optimization recommendations.
    """
    
    @staticmethod
    def detect_sparsity(tensor: torch.Tensor) -> float:
        """
        Detect the sparsity level of a tensor.

        Args:
            tensor: Input PyTorch tensor

        Returns:
            Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
        """
        if tensor.is_sparse:
            # For sparse tensors, calculate sparsity based on stored values
            total_elements = tensor.numel()
            non_zero_elements = tensor._nnz()
            return 1.0 - (non_zero_elements / total_elements)
        else:
            # For dense tensors, count zero elements
            total_elements = tensor.numel()
            if total_elements == 0:
                return 0.0
            zero_elements = torch.sum(tensor == 0).item()
            return zero_elements / total_elements

    @staticmethod
    def should_use_sparse(
        tensor: torch.Tensor, 
        sparsity_threshold: float = 0.5,
        memory_threshold: float = 0.8
    ) -> bool:
        """
        Determine if a tensor should use sparse representation based on sparsity and memory usage.

        Args:
            tensor: Input tensor
            sparsity_threshold: Minimum sparsity ratio to consider sparse conversion
            memory_threshold: Memory usage threshold (0.0-1.0) to trigger sparse conversion

        Returns:
            Boolean indicating if sparse representation should be used
        """
        sparsity = SparseTensorHandler.detect_sparsity(tensor)
        
        # Check sparsity threshold
        if sparsity < sparsity_threshold:
            return False
        
        # Check memory usage
        current_memory = MemoryOptimizer.get_tensor_memory_usage(tensor)
        available_memory = MemoryOptimizer.get_available_memory(str(tensor.device))
        
        if available_memory > 0:
            memory_usage_ratio = current_memory / available_memory
            if memory_usage_ratio > memory_threshold:
                log_info(f"Memory usage ratio {memory_usage_ratio:.3f} exceeds threshold {memory_threshold}")
                return True
        
        return sparsity >= sparsity_threshold

    @staticmethod
    def to_sparse(
        tensor: torch.Tensor, 
        threshold: float = 0.01
    ) -> torch.sparse.FloatTensor:
        """
        Convert a dense tensor to sparse representation with thresholding.

        Args:
            tensor: Dense PyTorch tensor
            threshold: Values below this threshold are considered zero

        Returns:
            Sparse tensor representation
        """
        if tensor.is_sparse:
            log_info("Tensor is already sparse")
            return tensor
        
        # Apply threshold to create true zeros
        tensor_thresholded = torch.where(
            torch.abs(tensor) < threshold, 
            torch.zeros_like(tensor), 
            tensor
        )
        
        # Convert to sparse COO format
        sparse_tensor = tensor_thresholded.to_sparse()
        
        original_memory = MemoryOptimizer.get_tensor_memory_usage(tensor)
        sparse_memory = MemoryOptimizer.get_tensor_memory_usage(sparse_tensor)
        memory_savings = original_memory - sparse_memory
        
        log_info(f"Converted tensor to sparse format. "
                f"Original size: {tensor.numel()}, "
                f"Sparse nnz: {sparse_tensor._nnz()}, "
                f"Memory savings: {memory_savings} bytes ({memory_savings/original_memory*100:.1f}%)")
        
        return sparse_tensor

    @staticmethod
    def to_dense(sparse_tensor: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Convert a sparse tensor to dense representation.

        Args:
            sparse_tensor: Sparse PyTorch tensor

        Returns:
            Dense tensor representation
        """
        if not sparse_tensor.is_sparse:
            log_info("Tensor is already dense")
            return sparse_tensor
        
        dense_tensor = sparse_tensor.to_dense()
        
        sparse_memory = MemoryOptimizer.get_tensor_memory_usage(sparse_tensor)
        dense_memory = MemoryOptimizer.get_tensor_memory_usage(dense_tensor)
        memory_increase = dense_memory - sparse_memory
        
        log_info(f"Converted sparse tensor to dense format. "
                f"Sparse nnz: {sparse_tensor._nnz()}, "
                f"Dense size: {dense_tensor.numel()}, "
                f"Memory increase: {memory_increase} bytes ({memory_increase/sparse_memory*100:.1f}%)")
        
        return dense_tensor

    @staticmethod
    def auto_convert(
        tensor: torch.Tensor,
        sparsity_threshold: float = 0.5,
        memory_threshold: float = 0.8,
        force_sparse: bool = False,
        force_dense: bool = False
    ) -> Union[torch.Tensor, torch.sparse.FloatTensor]:
        """
        Automatically convert between sparse and dense representations based on sparsity and memory usage.

        Args:
            tensor: Input tensor (dense or sparse)
            sparsity_threshold: Threshold for automatic conversion
            memory_threshold: Memory usage threshold for conversion
            force_sparse: Force conversion to sparse
            force_dense: Force conversion to dense

        Returns:
            Optimally represented tensor
        """
        if force_sparse and force_dense:
            raise ValueError("Cannot force both sparse and dense conversion")
        
        is_sparse = tensor.is_sparse
        
        if force_sparse and not is_sparse:
            return SparseTensorHandler.to_sparse(tensor)
        
        if force_dense and is_sparse:
            return SparseTensorHandler.to_dense(tensor)
        
        if is_sparse:
            # Check if we should convert to dense
            sparsity = SparseTensorHandler.detect_sparsity(tensor)
            if sparsity < sparsity_threshold:
                log_info(f"Converting sparse tensor to dense (sparsity: {sparsity:.3f})")
                return SparseTensorHandler.to_dense(tensor)
        else:
            # Check if we should convert to sparse
            if SparseTensorHandler.should_use_sparse(tensor, sparsity_threshold, memory_threshold):
                log_info(f"Converting dense tensor to sparse")
                return SparseTensorHandler.to_sparse(tensor)
        
        return tensor

    @staticmethod
    def get_sparse_info(tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Get comprehensive information about a tensor's sparsity characteristics.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary with sparsity information
        """
        info = {
            "is_sparse": tensor.is_sparse,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "total_elements": tensor.numel(),
        }
        
        if tensor.is_sparse:
            info.update({
                "stored_values": tensor._nnz(),
                "sparsity": SparseTensorHandler.detect_sparsity(tensor),
                "storage_format": "COO",  # PyTorch uses COO format by default
                "indices_shape": tuple(tensor._indices().shape),
                "values_shape": tuple(tensor._values().shape),
            })
        else:
            sparsity = SparseTensorHandler.detect_sparsity(tensor)
            info.update({
                "zero_elements": int(torch.sum(tensor == 0).item()),
                "non_zero_elements": int(torch.sum(tensor != 0).item()),
                "sparsity": sparsity,
                "should_be_sparse": SparseTensorHandler.should_use_sparse(tensor),
            })
        
        # Memory usage
        info["memory_usage_bytes"] = MemoryOptimizer.get_tensor_memory_usage(tensor)
        
        return info

    @staticmethod
    def compare_representations(tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compare memory usage and characteristics between sparse and dense representations.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary with comparison information
        """
        if tensor.is_sparse:
            dense_tensor = SparseTensorHandler.to_dense(tensor)
            sparse_tensor = tensor
        else:
            sparse_tensor = SparseTensorHandler.to_sparse(tensor)
            dense_tensor = tensor
        
        dense_memory = MemoryOptimizer.get_tensor_memory_usage(dense_tensor)
        sparse_memory = MemoryOptimizer.get_tensor_memory_usage(sparse_tensor)
        
        comparison = {
            "dense": {
                "memory_bytes": dense_memory,
                "elements": dense_tensor.numel(),
                "storage_efficiency": 1.0,
            },
            "sparse": {
                "memory_bytes": sparse_memory,
                "stored_values": sparse_tensor._nnz(),
                "storage_efficiency": sparse_memory / dense_memory if dense_memory > 0 else 0,
            },
            "memory_savings": {
                "absolute_bytes": dense_memory - sparse_memory,
                "percentage": ((dense_memory - sparse_memory) / dense_memory * 100) if dense_memory > 0 else 0,
            },
            "sparsity": SparseTensorHandler.detect_sparsity(dense_tensor),
            "recommendation": "sparse" if sparse_memory < dense_memory else "dense",
        }
        
        return comparison

    @staticmethod
    def optimize_tensor_storage(
        tensor: torch.Tensor, 
        threshold: float = 0.1,
        auto_convert: bool = True
    ) -> torch.Tensor:
        """
        Optimize tensor storage by choosing the best representation and applying compression.

        Args:
            tensor: Input tensor
            threshold: Sparsity threshold for conversion
            auto_convert: Whether to automatically convert between sparse/dense

        Returns:
            Optimized tensor
        """
        original_memory = MemoryOptimizer.get_tensor_memory_usage(tensor)
        
        # Auto-convert if enabled
        if auto_convert:
            tensor = SparseTensorHandler.auto_convert(tensor, sparsity_threshold=threshold)
        
        # Additional optimizations
        optimized_tensor = tensor
        
        # Check if we can reduce precision without significant loss
        if tensor.dtype == torch.float64:
            # Convert to float32 if the values don't require high precision
            if tensor.is_sparse:
                values = tensor._values()
                if torch.all(torch.abs(values) < 1e6):  # Arbitrary threshold for precision check
                    indices = tensor._indices()
                    values_f32 = values.float()
                    optimized_tensor = torch.sparse.FloatTensor(indices, values_f32, tensor.shape)
                    log_info("Reduced precision from float64 to float32")
            else:
                if torch.all(torch.abs(tensor) < 1e6):
                    optimized_tensor = tensor.float()
                    log_info("Reduced precision from float64 to float32")
        
        final_memory = MemoryOptimizer.get_tensor_memory_usage(optimized_tensor)
        memory_savings = original_memory - final_memory
        
        if memory_savings > 0:
            log_info(f"Storage optimization saved {memory_savings} bytes "
                    f"({memory_savings/original_memory*100:.1f}%)")
        
        return optimized_tensor

    @staticmethod
    def create_sparse_tensor(
        indices: torch.Tensor,
        values: torch.Tensor,
        shape: Tuple[int, ...],
        device: Optional[str] = None
    ) -> torch.sparse.FloatTensor:
        """
        Create a sparse tensor from indices and values.

        Args:
            indices: Tensor of indices (shape: [ndim, nnz])
            values: Tensor of values (shape: [nnz])
            shape: Shape of the resulting sparse tensor
            device: Target device

        Returns:
            Sparse tensor
        """
        if device is not None:
            indices = indices.to(device)
            values = values.to(device)
        
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        
        log_info(f"Created sparse tensor with shape {shape}, "
                f"nnz: {sparse_tensor._nnz()}, "
                f"sparsity: {SparseTensorHandler.detect_sparsity(sparse_tensor):.3f}")
        
        return sparse_tensor

    @staticmethod
    def serialize_sparse_tensor(tensor: torch.sparse.FloatTensor) -> Dict[str, Any]:
        """
        Serialize a sparse tensor for efficient storage.

        Args:
            tensor: Sparse tensor to serialize

        Returns:
            Dictionary with serialized tensor data
        """
        if not tensor.is_sparse:
            raise ValueError("Tensor must be sparse for sparse serialization")
        
        return {
            "indices": tensor._indices().cpu().numpy(),
            "values": tensor._values().cpu().numpy(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "nnz": tensor._nnz(),
            "sparsity": SparseTensorHandler.detect_sparsity(tensor),
        }

    @staticmethod
    def deserialize_sparse_tensor(
        data: Dict[str, Any], 
        device: Optional[str] = None
    ) -> torch.sparse.FloatTensor:
        """
        Deserialize a sparse tensor from stored data.

        Args:
            data: Dictionary with serialized tensor data
            device: Target device

        Returns:
            Reconstructed sparse tensor
        """
        indices = torch.from_numpy(data["indices"])
        values = torch.from_numpy(data["values"])
        shape = tuple(data["shape"])
        
        if device is not None:
            indices = indices.to(device)
            values = values.to(device)
        
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        
        log_info(f"Deserialized sparse tensor with shape {shape}, "
                f"nnz: {sparse_tensor._nnz()}")
        
        return sparse_tensor


class MemoryProfiler:
    """
    Advanced memory profiling and debugging tools for PyTorch tensors.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the memory profiler.

        Args:
            device: Device to profile
        """
        self.device = device
        self.snapshots = []
        self.operations = []
    
    def take_snapshot(self, name: str = ""):
        """
        Take a memory snapshot.

        Args:
            name: Optional name for the snapshot
        """
        snapshot = {
            "name": name,
            "timestamp": torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None,
            "memory_stats": MemoryOptimizer.get_memory_stats(self.device),
        }
        self.snapshots.append(snapshot)
        log_info(f"Memory snapshot '{name}' taken")
    
    def profile_operation(self, operation: callable, *args, **kwargs):
        """
        Profile memory usage during an operation.

        Args:
            operation: Function to profile
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, memory_profile)
        """
        self.take_snapshot("before_operation")
        
        result, memory_diff = MemoryOptimizer.monitor_memory_usage(
            operation, *args, device=self.device, **kwargs
        )
        
        self.take_snapshot("after_operation")
        
        profile = {
            "operation": operation.__name__ if hasattr(operation, '__name__') else str(operation),
            "memory_diff": memory_diff,
            "snapshots": self.snapshots[-2:],  # Last two snapshots
        }
        
        self.operations.append(profile)
        
        return result, profile
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive memory report.

        Returns:
            Dictionary with memory usage report
        """
        current_stats = MemoryOptimizer.get_memory_stats(self.device)
        
        report = {
            "device": self.device,
            "current_memory": current_stats,
            "snapshots": self.snapshots,
            "operations": self.operations,
            "total_snapshots": len(self.snapshots),
            "total_operations": len(self.operations),
        }
        
        if len(self.snapshots) > 1:
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            memory_change = (
                last_snapshot["memory_stats"]["allocated_memory"] - 
                first_snapshot["memory_stats"]["allocated_memory"]
            )
            
            report["session_memory_change"] = {
                "absolute_bytes": memory_change,
                "percentage": (memory_change / first_snapshot["memory_stats"]["allocated_memory"] * 100) 
                             if first_snapshot["memory_stats"]["allocated_memory"] > 0 else 0,
            }
        
        return report
    
    def clear_history(self):
        """Clear profiling history."""
        self.snapshots.clear()
        self.operations.clear()
        log_info("Memory profiler history cleared")


# Additional utility functions for sparse tensor operations
def batch_sparse_conversion(
    tensors: List[torch.Tensor],
    sparsity_threshold: float = 0.5,
    parallel: bool = True
) -> List[Union[torch.Tensor, torch.sparse.FloatTensor]]:
    """
    Convert a batch of tensors to optimal sparse/dense representation.

    Args:
        tensors: List of tensors to convert
        sparsity_threshold: Threshold for sparse conversion
        parallel: Whether to process tensors in parallel (if possible)

    Returns:
        List of optimally represented tensors
    """
    begin_section("Batch Sparse Conversion")
    
    converted_tensors = []
    total_memory_before = 0
    total_memory_after = 0
    
    for i, tensor in enumerate(tensors):
        memory_before = MemoryOptimizer.get_tensor_memory_usage(tensor)
        total_memory_before += memory_before
        
        converted = SparseTensorHandler.auto_convert(tensor, sparsity_threshold=sparsity_threshold)
        converted_tensors.append(converted)
        
        memory_after = MemoryOptimizer.get_tensor_memory_usage(converted)
        total_memory_after += memory_after
        
        log_info(f"Tensor {i}: {memory_before} -> {memory_after} bytes "
                f"({'sparse' if converted.is_sparse else 'dense'})")
    
    memory_savings = total_memory_before - total_memory_after
    log_info(f"Total memory savings: {memory_savings} bytes "
            f"({memory_savings/total_memory_before*100:.1f}%)")
    
    end_section()
    
    return converted_tensors


def create_sparse_dataset_loader(
    dataset_path: str,
    batch_size: int = 32,
    sparsity_threshold: float = 0.5,
    auto_convert: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a data loader that automatically handles sparse tensor conversion.

    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for the data loader
        sparsity_threshold: Threshold for automatic sparse conversion
        auto_convert: Whether to automatically convert tensors

    Returns:
        DataLoader with sparse tensor support
    """
    # This is a placeholder implementation
    # In practice, you would implement a custom Dataset class
    # that handles sparse tensor loading and conversion
    
    class SparseDataset(torch.utils.data.Dataset):
        def __init__(self, path, threshold, auto_convert):
            self.path = path
            self.threshold = threshold
            self.auto_convert = auto_convert
            # Load dataset metadata here
        
        def __len__(self):
            # Return dataset length
            return 1000  # Placeholder
        
        def __getitem__(self, idx):
            # Load and optionally convert tensor
            # This is a placeholder implementation
            tensor = torch.randn(64, 64, 64)  # Placeholder
            
            if self.auto_convert:
                tensor = SparseTensorHandler.auto_convert(tensor, self.threshold)
            
            return tensor
    
    dataset = SparseDataset(dataset_path, sparsity_threshold, auto_convert)
    
    def sparse_collate_fn(batch):
        """Custom collate function for sparse tensors."""
        # Handle batching of sparse tensors
        return torch.stack([item if not item.is_sparse else item.to_dense() for item in batch])
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=sparse_collate_fn if auto_convert else None,
        shuffle=True
    )


def optimize_model_for_sparse_tensors(model: nn.Module) -> nn.Module:
    """
    Optimize a model to work efficiently with sparse tensors.

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model
    """
    begin_section("Model Sparse Optimization")
    
    # Replace standard layers with sparse-aware versions where beneficial
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            # Consider replacing with SparseConv3d if the model processes sparse data
            log_info(f"Found Conv3d layer: {name} - consider SparseConv3d for sparse inputs")
        
        elif isinstance(module, nn.ConvTranspose3d):
            # Consider replacing with SparseConvTranspose3d
            log_info(f"Found ConvTranspose3d layer: {name} - consider SparseConvTranspose3d for sparse inputs")
        
        elif isinstance(module, nn.BatchNorm3d):
            # Consider replacing with SparseBatchNorm3d
            log_info(f"Found BatchNorm3d layer: {name} - consider SparseBatchNorm3d for sparse inputs")
    
    end_section()
    
    return model   def take_snapshot(self, name: str = None) -> Dict[str, Any]:
        """
        Take a memory snapshot.

        Args:
            name: Optional name for the snapshot

        Returns:
            Memory snapshot data
        """
        snapshot = {
            "timestamp": torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None,
            "name": name or f"snapshot_{len(self.snapshots)}",
            "memory_stats": MemoryOptimizer.get_memory_stats(self.device)
        }
        
        self.snapshots.append(snapshot)
        return snapshot

    def profile_operation(
        self,
        operation: callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Profile a specific operation.

        Args:
            operation: Function to profile
            operation_name: Name of the operation
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, profiling_data)
        """
        # Take before snapshot
        before_snapshot = self.take_snapshot(f"{operation_name}_before")
        
        # Run operation with timing
        if self.device.startswith("cuda") and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = operation(*args, **kwargs)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            import time
            start_time = time.time()
            result = operation(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Take after snapshot
        after_snapshot = self.take_snapshot(f"{operation_name}_after")
        
        # Calculate memory difference
        memory_diff = (
            after_snapshot["memory_stats"]["allocated_memory"] - 
            before_snapshot["memory_stats"]["allocated_memory"]
        )
        
        profile_data = {
            "operation_name": operation_name,
            "elapsed_time_ms": elapsed_time,
            "memory_increase": memory_diff,
            "before_memory": before_snapshot["memory_stats"]["allocated_memory"],
            "after_memory": after_snapshot["memory_stats"]["allocated_memory"],
        }
        
        self.operations.append(profile_data)
        
        return result, profile_data

    def generate_report(self) -> str:
        """
        Generate a comprehensive memory profiling report.

        Returns:
            Formatted report string
        """
        report = ["Memory Profiling Report", "=" * 50, ""]
        
        if self.snapshots:
            report.extend([
                "Memory Snapshots:",
                "-" * 20
            ])
            
            for snapshot in self.snapshots:
                stats = snapshot["memory_stats"]
                report.append(f"  {snapshot['name']}:")
                report.append(f"    Allocated: {stats['allocated_memory'] / (1024**2):.2f} MB")
                if "available_memory" in stats:
                    report.append(f"    Available: {stats['available_memory'] / (1024**2):.2f} MB")
                report.append("")
        
        if self.operations:
            report.extend([
                "Operation Profiles:",
                "-" * 20
            ])
            
            for op in self.operations:
                report.append(f"  {op['operation_name']}:")
                report.append(f"    Time: {op['elapsed_time_ms']:.2f} ms")
                report.append(f"    Memory increase: {op['memory_increase'] / (1024**2):.2f} MB")
                report.append("")
        
        return "\n".join(report)

    def clear_history(self):
        """Clear profiling history."""
        self.snapshots.clear()
        self.operations.clear()