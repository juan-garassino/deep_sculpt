"""
Data generation components for DeepSculpt PyTorch implementation.

This module provides the core data generation functionality, including
sculpture generation, shape creation, and data synthesis for training.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import random
from pathlib import Path

from .pytorch_sculptor import PyTorchSculptor
from .pytorch_shapes import ShapeType, SparseTensorHandler, PyTorchUtils


class DataGenerator:
    """
    Core data generator for creating 3D sculpture datasets.
    
    Handles the generation of individual samples and batches of 3D sculptures
    with configurable parameters and quality control.
    """
    
    def __init__(
        self,
        sculptor_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        sparse_mode: bool = False,
        quality_check: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize data generator.
        
        Args:
            sculptor_config: Configuration for PyTorchSculptor
            device: Device for tensor operations
            sparse_mode: Whether to use sparse tensors
            quality_check: Whether to perform quality checks on generated data
            seed: Random seed for reproducibility
        """
        self.device = device
        self.sparse_mode = sparse_mode
        self.quality_check = quality_check
        
        # Default sculptor configuration
        if sculptor_config is None:
            sculptor_config = {
                "void_dim": 64,
                "edges": (1, 0.3, 0.5),
                "planes": (1, 0.3, 0.5),
                "pipes": (1, 0.3, 0.5),
                "grid": (1, 4),
                "step": 1,
            }
        
        self.sculptor_config = sculptor_config
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Statistics tracking
        self.generation_stats = {
            "total_generated": 0,
            "generation_time": 0.0,
            "quality_failures": 0,
            "average_sparsity": 0.0,
        }
    
    def generate_single_sample(
        self,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single sculpture sample.
        
        Args:
            custom_config: Custom configuration for this sample
            
        Returns:
            Tuple of (structure_tensor, colors_tensor)
        """
        start_time = time.time()
        
        # Use custom config if provided, otherwise use default
        config = custom_config or self.sculptor_config
        
        # Create sculptor
        sculptor = PyTorchSculptor(
            device=self.device,
            sparse_mode=self.sparse_mode,
            verbose=False,
            **config
        )
        
        # Generate sculpture
        structure, colors = sculptor.generate_sculpture()
        
        # Quality check
        if self.quality_check:
            if not self._quality_check_sample(structure, colors):
                self.generation_stats["quality_failures"] += 1
                # Regenerate with different parameters
                return self.generate_single_sample(custom_config)
        
        # Update statistics
        generation_time = time.time() - start_time
        self.generation_stats["total_generated"] += 1
        self.generation_stats["generation_time"] += generation_time
        
        # Calculate sparsity
        sparsity = (structure == 0).float().mean().item()
        self.generation_stats["average_sparsity"] = (
            (self.generation_stats["average_sparsity"] * (self.generation_stats["total_generated"] - 1) + sparsity) /
            self.generation_stats["total_generated"]
        )
        
        return structure, colors
    
    def generate_batch(
        self,
        batch_size: int,
        custom_configs: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of sculpture samples.
        
        Args:
            batch_size: Number of samples to generate
            custom_configs: List of custom configurations for each sample
            
        Returns:
            Tuple of (batch_structures, batch_colors)
        """
        structures = []
        colors = []
        
        for i in range(batch_size):
            config = None
            if custom_configs and i < len(custom_configs):
                config = custom_configs[i]
            
            structure, color = self.generate_single_sample(config)
            structures.append(structure)
            colors.append(color)
        
        # Stack into batch tensors
        batch_structures = torch.stack(structures, dim=0)
        batch_colors = torch.stack(colors, dim=0)
        
        return batch_structures, batch_colors
    
    def generate_with_variations(
        self,
        base_config: Dict[str, Any],
        num_variations: int,
        variation_params: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate variations of a base configuration.
        
        Args:
            base_config: Base configuration to vary
            num_variations: Number of variations to generate
            variation_params: Parameters to vary with (min, max) ranges
            
        Returns:
            List of (structure, colors) tuples
        """
        samples = []
        
        for _ in range(num_variations):
            # Create varied configuration
            varied_config = base_config.copy()
            
            for param, (min_val, max_val) in variation_params.items():
                if param in varied_config:
                    # Generate random value in range
                    if isinstance(varied_config[param], tuple):
                        # Handle tuple parameters (like edges, planes, etc.)
                        original = varied_config[param]
                        varied = tuple(
                            val * random.uniform(min_val, max_val) if isinstance(val, (int, float))
                            else val for val in original
                        )
                        varied_config[param] = varied
                    else:
                        # Handle scalar parameters
                        varied_config[param] *= random.uniform(min_val, max_val)
            
            # Generate sample with varied configuration
            structure, colors = self.generate_single_sample(varied_config)
            samples.append((structure, colors))
        
        return samples
    
    def _quality_check_sample(
        self,
        structure: torch.Tensor,
        colors: torch.Tensor
    ) -> bool:
        """
        Perform quality check on generated sample.
        
        Args:
            structure: Structure tensor
            colors: Colors tensor
            
        Returns:
            True if sample passes quality check
        """
        # Check for NaN or Inf values
        if torch.isnan(structure).any() or torch.isinf(structure).any():
            return False
        if torch.isnan(colors).any() or torch.isinf(colors).any():
            return False
        
        # Check for reasonable sparsity (not completely empty or full)
        sparsity = (structure == 0).float().mean().item()
        if sparsity < 0.1 or sparsity > 0.95:
            return False
        
        # Check for reasonable value ranges
        if structure.min() < 0 or structure.max() > 1:
            return False
        if colors.min() < 0 or colors.max() > 1:
            return False
        
        # Check for minimum complexity (some variation in structure)
        if structure.std() < 0.01:
            return False
        
        return True
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.generation_stats.copy()
        if stats["total_generated"] > 0:
            stats["average_generation_time"] = stats["generation_time"] / stats["total_generated"]
            stats["quality_failure_rate"] = stats["quality_failures"] / stats["total_generated"]
        else:
            stats["average_generation_time"] = 0.0
            stats["quality_failure_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            "total_generated": 0,
            "generation_time": 0.0,
            "quality_failures": 0,
            "average_sparsity": 0.0,
        }


class ParametricDataGenerator(DataGenerator):
    """
    Parametric data generator with controlled parameter sampling.
    
    Allows for systematic exploration of the parameter space and
    generation of datasets with specific parameter distributions.
    """
    
    def __init__(
        self,
        parameter_ranges: Dict[str, Dict[str, Any]],
        sampling_strategy: str = "uniform",
        **kwargs
    ):
        """
        Initialize parametric data generator.
        
        Args:
            parameter_ranges: Dictionary defining parameter ranges and distributions
            sampling_strategy: Strategy for sampling parameters ("uniform", "normal", "grid")
            **kwargs: Additional arguments for base DataGenerator
        """
        super().__init__(**kwargs)
        self.parameter_ranges = parameter_ranges
        self.sampling_strategy = sampling_strategy
        
        # Validate parameter ranges
        self._validate_parameter_ranges()
    
    def _validate_parameter_ranges(self):
        """Validate parameter range specifications."""
        required_keys = {"min", "max"}
        for param, spec in self.parameter_ranges.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Parameter range for {param} must be a dictionary")
            if not required_keys.issubset(spec.keys()):
                raise ValueError(f"Parameter range for {param} must contain 'min' and 'max' keys")
    
    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameters according to the specified strategy.
        
        Returns:
            Dictionary of sampled parameters
        """
        sampled_params = {}
        
        for param, spec in self.parameter_ranges.items():
            min_val = spec["min"]
            max_val = spec["max"]
            
            if self.sampling_strategy == "uniform":
                if isinstance(min_val, int) and isinstance(max_val, int):
                    sampled_params[param] = random.randint(min_val, max_val)
                else:
                    sampled_params[param] = random.uniform(min_val, max_val)
            
            elif self.sampling_strategy == "normal":
                mean = spec.get("mean", (min_val + max_val) / 2)
                std = spec.get("std", (max_val - min_val) / 6)  # 3-sigma rule
                value = np.random.normal(mean, std)
                value = np.clip(value, min_val, max_val)
                sampled_params[param] = value
            
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        return sampled_params
    
    def generate_parametric_sample(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Generate a sample with sampled parameters.
        
        Returns:
            Tuple of (structure, colors, parameters_used)
        """
        # Sample parameters
        sampled_params = self.sample_parameters()
        
        # Create configuration with sampled parameters
        config = self.sculptor_config.copy()
        config.update(sampled_params)
        
        # Generate sample
        structure, colors = self.generate_single_sample(config)
        
        return structure, colors, sampled_params
    
    def generate_parametric_dataset(
        self,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate a parametric dataset.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (structures, colors, parameters_list)
        """
        structures = []
        colors = []
        parameters_list = []
        
        for _ in range(num_samples):
            structure, color, params = self.generate_parametric_sample()
            structures.append(structure)
            colors.append(color)
            parameters_list.append(params)
        
        # Stack into batch tensors
        batch_structures = torch.stack(structures, dim=0)
        batch_colors = torch.stack(colors, dim=0)
        
        return batch_structures, batch_colors, parameters_list


class ConditionalDataGenerator(DataGenerator):
    """
    Conditional data generator for creating samples with specific conditions.
    
    Supports generation of samples conditioned on various attributes like
    complexity, size, shape type, etc.
    """
    
    def __init__(
        self,
        condition_mappings: Dict[str, Dict[str, Any]],
        **kwargs
    ):
        """
        Initialize conditional data generator.
        
        Args:
            condition_mappings: Mapping from condition values to parameter configurations
            **kwargs: Additional arguments for base DataGenerator
        """
        super().__init__(**kwargs)
        self.condition_mappings = condition_mappings
        self.available_conditions = list(condition_mappings.keys())
    
    def generate_conditional_sample(
        self,
        condition: str,
        condition_value: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a sample with specific condition.
        
        Args:
            condition: Type of condition (e.g., "complexity", "size")
            condition_value: Value of the condition
            
        Returns:
            Tuple of (structure, colors)
        """
        if condition not in self.condition_mappings:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Get configuration for this condition
        condition_config = self.condition_mappings[condition].get(condition_value)
        if condition_config is None:
            raise ValueError(f"Unknown condition value: {condition_value} for condition: {condition}")
        
        # Merge with base configuration
        config = self.sculptor_config.copy()
        config.update(condition_config)
        
        return self.generate_single_sample(config)
    
    def generate_conditional_batch(
        self,
        conditions: List[Tuple[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch with specific conditions.
        
        Args:
            conditions: List of (condition_type, condition_value) tuples
            
        Returns:
            Tuple of (batch_structures, batch_colors)
        """
        structures = []
        colors = []
        
        for condition_type, condition_value in conditions:
            structure, color = self.generate_conditional_sample(condition_type, condition_value)
            structures.append(structure)
            colors.append(color)
        
        # Stack into batch tensors
        batch_structures = torch.stack(structures, dim=0)
        batch_colors = torch.stack(colors, dim=0)
        
        return batch_structures, batch_colors
    
    def get_available_conditions(self) -> Dict[str, List[Any]]:
        """Get available conditions and their possible values."""
        return {
            condition: list(mappings.keys())
            for condition, mappings in self.condition_mappings.items()
        }


# Convenience functions
def create_simple_generator(device: str = "cuda", void_dim: int = 64) -> DataGenerator:
    """Create a simple data generator with default settings."""
    config = {
        "void_dim": void_dim,
        "edges": (1, 0.3, 0.5),
        "planes": (1, 0.3, 0.5),
        "pipes": (1, 0.3, 0.5),
        "grid": (1, 4),
        "step": 1,
    }
    return DataGenerator(sculptor_config=config, device=device)


def create_parametric_generator(
    device: str = "cuda",
    void_dim_range: Tuple[int, int] = (32, 128),
    complexity_range: Tuple[float, float] = (0.1, 0.8)
) -> ParametricDataGenerator:
    """Create a parametric data generator with common parameter ranges."""
    parameter_ranges = {
        "void_dim": {"min": void_dim_range[0], "max": void_dim_range[1]},
        "edges": {"min": (1, 0.1, complexity_range[0]), "max": (3, 0.5, complexity_range[1])},
        "planes": {"min": (1, 0.1, complexity_range[0]), "max": (2, 0.4, complexity_range[1])},
        "pipes": {"min": (1, 0.2, complexity_range[0]), "max": (2, 0.6, complexity_range[1])},
    }
    
    return ParametricDataGenerator(
        parameter_ranges=parameter_ranges,
        device=device,
        sampling_strategy="uniform"
    )