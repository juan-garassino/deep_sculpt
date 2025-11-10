"""
PyTorch-based Dataset Preprocessing System for DeepSculpt

This module provides PyTorch-based preprocessing of sculpture datasets for machine learning tasks.
It migrates the original TensorFlow-based curator.py to use PyTorch tensors and operations,
providing various encoding methods (one-hot, binary, RGB), efficient batch processing,
and GPU acceleration support.

Key features:
- PyTorch tensor-based operations with GPU support
- Multiple encoding methods: One-hot, binary, RGB, and custom embeddings
- Memory-efficient batch processing and streaming
- Automatic sparse/dense tensor handling
- Backward compatibility with original TensorFlow implementation
- Advanced data augmentation and transformation pipelines

Dependencies:
- torch: For tensor operations and GPU acceleration
- numpy: For array operations and compatibility
- sklearn: For label encoding utilities
- matplotlib: For color mapping in RGB encoding
- tqdm: For progress visualization

Used by:
- PyTorch training scripts: For preparing data for PyTorch models
- Model evaluation: For processing inference results
- Data streaming pipelines: For efficient large dataset handling
"""

import os
import time
import random
import glob
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from .logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)


class PyTorchEncoderDecoder:
    """Base class for PyTorch-based encoding and decoding methods."""

    def __init__(
        self,
        colors: Union[np.ndarray, torch.Tensor],
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        Initialize the PyTorchEncoderDecoder.

        Args:
            colors: Array/tensor of colors to encode
            device: Device to use for computations ('cuda' or 'cpu')
            verbose: Whether to print detailed information
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        
        # Store original colors array (may contain objects/strings)
        if isinstance(colors, np.ndarray):
            self.colors = colors
        else:
            self.colors = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else colors
            
        self.unique_colors = self._get_unique_colors()

    def _get_unique_colors(self) -> Set[Any]:
        """
        Extract all unique colors from the colors array.

        Returns:
            Set of unique color values
        """
        unique_colors = set()
        # Flatten the array and add each unique color to the set
        flat_colors = self.colors.flatten()
        for color in flat_colors:
            unique_colors.add(color)

        if self.verbose:
            log_info(f"Found {len(unique_colors)} unique colors: {unique_colors}")

        return unique_colors


class PyTorchOneHotEncoderDecoder(PyTorchEncoderDecoder):
    """
    PyTorch-based class for one-hot encoding and decoding color values.
    """

    def __init__(
        self,
        colors_tensor: Union[np.ndarray, torch.Tensor],
        color_list: Optional[List[Any]] = None,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Initialize the PyTorchOneHotEncoderDecoder.

        Args:
            colors_tensor: Tensor of colors to encode
            color_list: List of all possible color values (if None, extracted from data)
            device: Device to use for computations
            verbose: Whether to print detailed information
        """
        super().__init__(colors_tensor, device, verbose)
        
        # Store the original colors array for processing
        if isinstance(colors_tensor, np.ndarray):
            self.colors_tensor = colors_tensor
        else:
            self.colors_tensor = colors_tensor.cpu().numpy() if isinstance(colors_tensor, torch.Tensor) else colors_tensor
            
        # Handle different array shapes
        if len(self.colors_tensor.shape) == 4:  # (samples, dim, dim, dim)
            self.void_dim = self.colors_tensor.shape[1]
            self.n_samples = self.colors_tensor.shape[0]
            self.original_shape = self.colors_tensor.shape
        elif len(self.colors_tensor.shape) == 3:  # (dim, dim, dim) - single sample or (samples, dim, dim)
            if self.colors_tensor.shape[0] == 1:  # Single sample case
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = 1
                self.original_shape = self.colors_tensor.shape
            else:  # Multiple 2D samples
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = self.colors_tensor.shape[0]
                self.original_shape = self.colors_tensor.shape
        else:
            raise ValueError(f"Unsupported colors tensor shape: {self.colors_tensor.shape}")
            
        self.n_classes = None
        self.classes = None

        # Use provided color list or determine from data
        if color_list is not None:
            self.color_list = color_list
        else:
            self.color_list = sorted(list(self.unique_colors), key=lambda x: str(x))

        # Create color to index mapping
        self.color_to_idx = {color: idx for idx, color in enumerate(self.color_list)}
        self.idx_to_color = {idx: color for color, idx in self.color_to_idx.items()}
        self.n_classes = len(self.color_list)

    def ohe_encode(self) -> Tuple[torch.Tensor, List[Any]]:
        """
        Encode colors using one-hot encoding with PyTorch operations.

        Returns:
            Tuple of (encoded_tensor, color_classes)
        """
        begin_section("PyTorch One-Hot Encoding Colors")

        try:
            if not self.color_list:
                raise ValueError("The list of colors cannot be empty.")

            # Convert colors to indices
            colors_flat = self.colors_tensor.flatten()
            indices = np.array([self.color_to_idx.get(color, 0) for color in colors_flat])
            indices_tensor = torch.from_numpy(indices).long().to(self.device)

            # Create one-hot encoding using PyTorch
            one_hot_flat = F.one_hot(indices_tensor, num_classes=self.n_classes).float()

            # Reshape back to original dimensions plus one-hot dimension
            if len(self.original_shape) == 4:  # 3D voxels
                encoded_tensor = one_hot_flat.reshape(
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_classes
                )
            elif len(self.original_shape) == 3:  # 2D case
                encoded_tensor = one_hot_flat.reshape(
                    self.original_shape[0],
                    self.original_shape[1],
                    self.original_shape[2],
                    self.n_classes
                )

            self.classes = self.color_list

            log_info(f"Encoded {self.n_samples} samples into {self.n_classes} classes")
            log_info(f"Classes: {self.classes}")
            log_success(f"One-hot encoded to shape {encoded_tensor.shape}")
            end_section()

            return encoded_tensor, self.classes

        except Exception as e:
            log_error(f"Error during PyTorch one-hot encoding: {str(e)}")
            end_section("PyTorch one-hot encoding failed")
            raise

    def ohe_decode(
        self, one_hot_encoded_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode one-hot encoded colors back to original values using PyTorch.

        Args:
            one_hot_encoded_tensor: One-hot encoded tensor

        Returns:
            Tuple of (structures_tensor, colors_tensor) where:
              - structures_tensor has 1s where material exists and 0s elsewhere
              - colors_tensor contains the original color indices
        """
        begin_section("PyTorch One-Hot Decoding Colors")

        try:
            # Get indices from one-hot encoding
            indices_tensor = torch.argmax(one_hot_encoded_tensor, dim=-1)
            
            # Create structures tensor (1 where material exists, 0 elsewhere)
            # Assuming index 0 corresponds to None/empty
            structures_tensor = (indices_tensor != 0).float()
            
            # Convert indices back to colors (keeping as indices for efficiency)
            colors_tensor = indices_tensor.float()

            log_success(
                f"Decoded to shapes: structures {structures_tensor.shape}, colors {colors_tensor.shape}"
            )
            end_section()

            return structures_tensor, colors_tensor

        except Exception as e:
            log_error(f"Error during PyTorch one-hot decoding: {str(e)}")
            end_section("PyTorch one-hot decoding failed")
            raise


class PyTorchBinaryEncoderDecoder(PyTorchEncoderDecoder):
    """
    PyTorch-based class for binary encoding and decoding color values.
    """

    def __init__(
        self,
        colors_tensor: Union[np.ndarray, torch.Tensor],
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        Initialize the PyTorchBinaryEncoderDecoder.

        Args:
            colors_tensor: Tensor of colors to encode
            device: Device to use for computations
            verbose: Whether to print detailed information
        """
        super().__init__(colors_tensor, device, verbose)
        
        # Store the original colors array for processing
        if isinstance(colors_tensor, np.ndarray):
            self.colors_tensor = colors_tensor
        else:
            self.colors_tensor = colors_tensor.cpu().numpy() if isinstance(colors_tensor, torch.Tensor) else colors_tensor
            
        # Handle different array shapes
        if len(self.colors_tensor.shape) == 4:  # (samples, dim, dim, dim)
            self.void_dim = self.colors_tensor.shape[1]
            self.n_samples = self.colors_tensor.shape[0]
            self.original_shape = self.colors_tensor.shape
        elif len(self.colors_tensor.shape) == 3:  # (dim, dim, dim) - single sample or (samples, dim, dim)
            if self.colors_tensor.shape[0] == 1:  # Single sample case
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = 1
                self.original_shape = self.colors_tensor.shape
            else:  # Multiple 2D samples
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = self.colors_tensor.shape[0]
                self.original_shape = self.colors_tensor.shape
        else:
            raise ValueError(f"Unsupported colors tensor shape: {self.colors_tensor.shape}")
            
        self.classes = None
        self.n_bit = None
        
        # Use sklearn's LabelEncoder for consistency with original implementation
        self.label_encoder = LabelEncoder()

    def binary_encode(self) -> Tuple[torch.Tensor, List[Any]]:
        """
        Encode colors using binary encoding with PyTorch operations.

        Returns:
            Tuple of (encoded_tensor, color_classes)
        """
        begin_section("PyTorch Binary Encoding Colors")

        try:
            # Flatten the array for label encoding
            flat_colors = self.colors_tensor.flatten()

            # Transform using sklearn's LabelEncoder
            label_encoded = self.label_encoder.fit_transform(flat_colors)

            # Get the classes from the encoder
            self.classes = self.label_encoder.classes_

            # Calculate how many bits we need to represent all classes
            self.n_bit = int(np.ceil(np.log2(len(self.classes))))

            log_info(f"Encoding {len(self.classes)} classes using {self.n_bit} bits")

            # Convert to PyTorch tensor
            label_tensor = torch.from_numpy(label_encoded).long().to(self.device)

            # Convert each label to its binary representation using PyTorch
            # Create a tensor to hold binary representations
            binary_tensor = torch.zeros(
                len(label_encoded), self.n_bit, device=self.device, dtype=torch.float32
            )

            # Convert to binary using bit operations
            for bit in range(self.n_bit):
                binary_tensor[:, bit] = (label_tensor >> bit) & 1

            # Reshape to original dimensions plus bit dimension
            if len(self.original_shape) == 4:  # 3D voxels
                binary_encoded_tensor = binary_tensor.reshape(
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_bit
                )
            elif len(self.original_shape) == 3:  # 2D case
                binary_encoded_tensor = binary_tensor.reshape(
                    self.original_shape[0],
                    self.original_shape[1],
                    self.original_shape[2],
                    self.n_bit
                )

            log_success(f"Binary encoded to shape {binary_encoded_tensor.shape}")
            end_section()

            return binary_encoded_tensor, list(self.classes)

        except Exception as e:
            log_error(f"Error during PyTorch binary encoding: {str(e)}")
            end_section("PyTorch binary encoding failed")
            raise

    def binary_decode(
        self, binary_encoded_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode binary encoded colors back to original values using PyTorch.

        Args:
            binary_encoded_tensor: Binary encoded tensor

        Returns:
            Tuple of (structures_tensor, colors_tensor) where:
              - structures_tensor has 1s where material exists and 0s elsewhere
              - colors_tensor contains the original color indices
        """
        begin_section("PyTorch Binary Decoding Colors")

        try:
            # Reshape the tensor for decoding
            flat_encoded = binary_encoded_tensor.reshape(-1, self.n_bit)

            # Convert binary vectors back to integers using PyTorch
            powers_of_2 = torch.pow(2, torch.arange(self.n_bit, device=self.device)).float()
            label_indices = torch.sum(flat_encoded * powers_of_2, dim=1).long()

            # Convert to CPU for sklearn inverse transform
            label_indices_cpu = label_indices.cpu().numpy()
            
            # Convert integer labels back to original classes
            decoded_colors_np = self.label_encoder.inverse_transform(label_indices_cpu)
            
            # Convert back to tensor indices for efficiency
            color_to_idx = {color: idx for idx, color in enumerate(self.classes)}
            color_indices = np.array([color_to_idx.get(color, 0) for color in decoded_colors_np])
            colors_tensor = torch.from_numpy(color_indices).float().to(self.device)

            # Create structures tensor (1 where material exists, 0 elsewhere)
            # Assuming index 0 corresponds to None/empty
            structures_tensor = (colors_tensor != 0).float()

            # Reshape back to original dimensions
            structures_tensor = structures_tensor.reshape(
                self.n_samples, self.void_dim, self.void_dim, self.void_dim
            )
            colors_tensor = colors_tensor.reshape(
                self.n_samples, self.void_dim, self.void_dim, self.void_dim
            )

            log_success(
                f"Decoded to shapes: structures {structures_tensor.shape}, colors {colors_tensor.shape}"
            )
            end_section()

            return structures_tensor, colors_tensor

        except Exception as e:
            log_error(f"Error during PyTorch binary decoding: {str(e)}")
            end_section("PyTorch binary decoding failed")
            raise


class PyTorchRGBEncoderDecoder(PyTorchEncoderDecoder):
    """
    PyTorch-based class for RGB encoding and decoding color values.
    """

    def __init__(
        self,
        colors_tensor: Optional[Union[np.ndarray, torch.Tensor]] = None,
        color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Initialize the PyTorchRGBEncoderDecoder.

        Args:
            colors_tensor: Tensor of colors to encode (optional)
            color_dict: Dictionary mapping color names to RGB tuples (optional)
            device: Device to use for computations
            verbose: Whether to print detailed information
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        
        if colors_tensor is not None:
            # Store the original colors array for processing
            if isinstance(colors_tensor, np.ndarray):
                self.colors_tensor = colors_tensor
            else:
                self.colors_tensor = colors_tensor.cpu().numpy() if isinstance(colors_tensor, torch.Tensor) else colors_tensor
            
            # Handle different array shapes
            if len(self.colors_tensor.shape) == 4:  # (samples, dim, dim, dim)
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = self.colors_tensor.shape[0]
                self.original_shape = self.colors_tensor.shape
            elif len(self.colors_tensor.shape) == 3:  # (dim, dim, dim) - single sample or (samples, dim, dim)
                if self.colors_tensor.shape[0] == 1:  # Single sample case
                    self.void_dim = self.colors_tensor.shape[1]
                    self.n_samples = 1
                    self.original_shape = self.colors_tensor.shape
                else:  # Multiple 2D samples
                    self.void_dim = self.colors_tensor.shape[1]
                    self.n_samples = self.colors_tensor.shape[0]
                    self.original_shape = self.colors_tensor.shape
            else:
                raise ValueError(f"Unsupported colors tensor shape: {self.colors_tensor.shape}")
            
            super().__init__(colors_tensor, device, verbose)
        else:
            self.colors_tensor = None
            self.void_dim = 0
            self.n_samples = 0
            self.original_shape = None

        # Initialize color dictionary
        self.color_dict = self._initialize_color_dict(color_dict)

    def _initialize_color_dict(
        self, color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None
    ) -> Dict[Any, Tuple[int, int, int]]:
        """
        Initialize the color dictionary with defaults if not provided.

        Args:
            color_dict: Dictionary mapping color names to RGB tuples

        Returns:
            Complete color dictionary
        """
        if color_dict is not None:
            return color_dict

        # Create default color dictionary using matplotlib colors
        result_dict = {}

        # Add TABLEAU colors (common visualization colors)
        for name, hex_color in mcolors.TABLEAU_COLORS.items():
            rgb = tuple(int(x * 255) for x in mcolors.to_rgb(hex_color))
            result_dict[name] = rgb

        # Add CSS4 colors (wide range of named colors)
        for name, hex_color in mcolors.CSS4_COLORS.items():
            if name not in result_dict:  # Don't overwrite TABLEAU colors
                rgb = tuple(int(x * 255) for x in mcolors.to_rgb(hex_color))
                result_dict[name] = rgb

        # Add special case for None (transparent/empty)
        result_dict[None] = (0, 0, 0)

        # Add common color names that might be used
        common_colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }

        for name, rgb in common_colors.items():
            if name not in result_dict:
                result_dict[name] = rgb

        if self.verbose:
            log_info(f"Initialized color dictionary with {len(result_dict)} colors")

        return result_dict

    def rgb_encode(self) -> Tuple[torch.Tensor, Dict[Any, Tuple[int, int, int]]]:
        """
        Encode colors tensor using RGB values with PyTorch operations.

        Returns:
            Tuple of (encoded_tensor, color_mapping)
        """
        begin_section("PyTorch RGB Encoding Colors")

        try:
            if self.colors_tensor is None:
                raise ValueError("Colors tensor not provided")

            # Initialize output tensor with RGB channels
            if len(self.original_shape) == 4:  # 3D voxels
                rgb_tensor = torch.zeros(
                    (self.n_samples, self.void_dim, self.void_dim, self.void_dim, 3),
                    dtype=torch.uint8,
                    device=self.device
                )
            elif len(self.original_shape) == 3:  # 2D case
                rgb_tensor = torch.zeros(
                    (self.original_shape[0], self.original_shape[1], self.original_shape[2], 3),
                    dtype=torch.uint8,
                    device=self.device
                )

            # Use colors array directly (already numpy)
            colors_cpu = self.colors_tensor

            # Convert each color to its RGB value
            if len(self.original_shape) == 4:  # 3D voxels
                for s in range(self.n_samples):
                    for i in range(self.void_dim):
                        for j in range(self.void_dim):
                            for k in range(self.void_dim):
                                color = colors_cpu[s, i, j, k]
                                if color in self.color_dict:
                                    rgb_tensor[s, i, j, k] = torch.tensor(
                                        self.color_dict[color], device=self.device
                                    )
                                elif color is not None:
                                    # Assign a default gray for unknown colors
                                    rgb_tensor[s, i, j, k] = torch.tensor(
                                        (128, 128, 128), device=self.device
                                    )
            elif len(self.original_shape) == 3:  # 2D case
                for s in range(self.original_shape[0]):
                    for i in range(self.original_shape[1]):
                        for j in range(self.original_shape[2]):
                            color = colors_cpu[s, i, j]
                            if color in self.color_dict:
                                rgb_tensor[s, i, j] = torch.tensor(
                                    self.color_dict[color], device=self.device
                                )
                            elif color is not None:
                                # Assign a default gray for unknown colors
                                rgb_tensor[s, i, j] = torch.tensor(
                                    (128, 128, 128), device=self.device
                                )

            log_success(f"RGB encoded to shape {rgb_tensor.shape}")
            end_section()

            return rgb_tensor, self.color_dict

        except Exception as e:
            log_error(f"Error during PyTorch RGB encoding: {str(e)}")
            end_section("PyTorch RGB encoding failed")
            raise

    def rgb_decode(
        self, rgb_tensor: torch.Tensor, threshold: float = 10.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode RGB encoded tensor back to colors using PyTorch operations.

        Args:
            rgb_tensor: RGB encoded tensor
            threshold: Threshold for color matching (Euclidean distance)

        Returns:
            Tuple of (structures_tensor, colors_tensor) where:
              - structures_tensor has 1s where material exists and 0s elsewhere
              - colors_tensor contains the original color indices
        """
        begin_section("PyTorch RGB Decoding Colors")

        try:
            n_samples = rgb_tensor.shape[0]
            void_dim = rgb_tensor.shape[1]

            # Prepare output tensors
            colors_tensor = torch.zeros(
                (n_samples, void_dim, void_dim, void_dim),
                dtype=torch.long,
                device=self.device
            )
            structures_tensor = torch.zeros(
                (n_samples, void_dim, void_dim, void_dim),
                dtype=torch.float32,
                device=self.device
            )

            # Create reverse mapping and RGB lookup tensors
            color_list = list(self.color_dict.keys())
            rgb_list = [self.color_dict[color] for color in color_list]
            rgb_lookup = torch.tensor(rgb_list, dtype=torch.float32, device=self.device)

            # Flatten RGB tensor for vectorized operations
            rgb_flat = rgb_tensor.reshape(-1, 3).float()

            # Compute distances to all known colors
            distances = torch.cdist(rgb_flat.unsqueeze(0), rgb_lookup.unsqueeze(0)).squeeze(0)
            
            # Find closest colors
            min_distances, closest_indices = torch.min(distances, dim=1)
            
            # Apply threshold
            valid_matches = min_distances <= threshold
            
            # Assign colors
            color_indices = torch.zeros_like(closest_indices)
            color_indices[valid_matches] = closest_indices[valid_matches]
            
            # Reshape back
            colors_tensor = color_indices.reshape(n_samples, void_dim, void_dim, void_dim)
            
            # Create structures tensor (1 where material exists, 0 elsewhere)
            # Assuming index 0 corresponds to None/empty
            structures_tensor = (colors_tensor != 0).float()

            log_success(
                f"RGB decoded to shapes: structures {structures_tensor.shape}, colors {colors_tensor.shape}"
            )
            end_section()

            return structures_tensor, colors_tensor

        except Exception as e:
            log_error(f"Error during PyTorch RGB decoding: {str(e)}")
            end_section("PyTorch RGB decoding failed")
            raise


class PyTorchEmbeddingEncoderDecoder(PyTorchEncoderDecoder):
    """
    PyTorch-based class for learned embedding encoding and decoding.
    """

    def __init__(
        self,
        colors_tensor: Union[np.ndarray, torch.Tensor],
        embedding_dim: int = 64,
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        Initialize the PyTorchEmbeddingEncoderDecoder.

        Args:
            colors_tensor: Tensor of colors to encode
            embedding_dim: Dimension of learned embeddings
            device: Device to use for computations
            verbose: Whether to print detailed information
        """
        super().__init__(colors_tensor, device, verbose)
        
        # Store the original colors array for processing
        if isinstance(colors_tensor, np.ndarray):
            self.colors_tensor = colors_tensor
        else:
            self.colors_tensor = colors_tensor.cpu().numpy() if isinstance(colors_tensor, torch.Tensor) else colors_tensor
            
        # Handle different array shapes
        if len(self.colors_tensor.shape) == 4:  # (samples, dim, dim, dim)
            self.void_dim = self.colors_tensor.shape[1]
            self.n_samples = self.colors_tensor.shape[0]
            self.original_shape = self.colors_tensor.shape
        elif len(self.colors_tensor.shape) == 3:  # (dim, dim, dim) - single sample or (samples, dim, dim)
            if self.colors_tensor.shape[0] == 1:  # Single sample case
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = 1
                self.original_shape = self.colors_tensor.shape
            else:  # Multiple 2D samples
                self.void_dim = self.colors_tensor.shape[1]
                self.n_samples = self.colors_tensor.shape[0]
                self.original_shape = self.colors_tensor.shape
        else:
            raise ValueError(f"Unsupported colors tensor shape: {self.colors_tensor.shape}")
            
        self.embedding_dim = embedding_dim
        
        # Create color to index mapping
        self.color_list = sorted(list(self.unique_colors), key=lambda x: str(x))
        self.color_to_idx = {color: idx for idx, color in enumerate(self.color_list)}
        self.n_classes = len(self.color_list)
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(self.n_classes, embedding_dim).to(self.device)

    def embedding_encode(self) -> Tuple[torch.Tensor, nn.Embedding]:
        """
        Encode colors using learned embeddings.

        Returns:
            Tuple of (encoded_tensor, embedding_layer)
        """
        begin_section("PyTorch Embedding Encoding Colors")

        try:
            # Convert colors to indices
            colors_flat = self.colors_tensor.flatten()
            indices = np.array([self.color_to_idx.get(color, 0) for color in colors_flat])
            indices_tensor = torch.from_numpy(indices).long().to(self.device)

            # Apply embedding
            embedded_flat = self.embedding(indices_tensor)

            # Reshape back to original dimensions plus embedding dimension
            if len(self.original_shape) == 4:  # 3D voxels
                encoded_tensor = embedded_flat.reshape(
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.embedding_dim
                )
            elif len(self.original_shape) == 3:  # 2D case
                encoded_tensor = embedded_flat.reshape(
                    self.original_shape[0],
                    self.original_shape[1],
                    self.original_shape[2],
                    self.embedding_dim
                )

            log_success(f"Embedding encoded to shape {encoded_tensor.shape}")
            end_section()

            return encoded_tensor, self.embedding

        except Exception as e:
            log_error(f"Error during PyTorch embedding encoding: {str(e)}")
            end_section("PyTorch embedding encoding failed")
            raise

    def embedding_decode(
        self, embedded_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode embedded colors back to indices using nearest neighbor search.

        Args:
            embedded_tensor: Embedded tensor

        Returns:
            Tuple of (structures_tensor, colors_tensor)
        """
        begin_section("PyTorch Embedding Decoding Colors")

        try:
            # Get embedding weights
            embedding_weights = self.embedding.weight  # Shape: (n_classes, embedding_dim)
            
            # Flatten embedded tensor
            embedded_flat = embedded_tensor.reshape(-1, self.embedding_dim)
            
            # Compute distances to all embedding vectors
            distances = torch.cdist(embedded_flat, embedding_weights)
            
            # Find closest embeddings
            _, closest_indices = torch.min(distances, dim=1)
            
            # Reshape back
            if len(self.original_shape) == 4:  # 3D voxels
                colors_tensor = closest_indices.reshape(
                    self.n_samples, self.void_dim, self.void_dim, self.void_dim
                ).float()
            elif len(self.original_shape) == 3:  # 2D case
                colors_tensor = closest_indices.reshape(
                    self.original_shape[0], self.original_shape[1], self.original_shape[2]
                ).float()
            
            # Create structures tensor
            structures_tensor = (colors_tensor != 0).float()

            log_success(
                f"Embedding decoded to shapes: structures {structures_tensor.shape}, colors {colors_tensor.shape}"
            )
            end_section()

            return structures_tensor, colors_tensor

        except Exception as e:
            log_error(f"Error during PyTorch embedding decoding: {str(e)}")
            end_section("PyTorch embedding decoding failed")
            raise


class PyTorchDataset(Dataset):
    """
    PyTorch Dataset class for efficient data loading and preprocessing.
    """

    def __init__(
        self,
        data_paths: List[Tuple[str, str]],
        encoder_decoder: PyTorchEncoderDecoder,
        transform: Optional[Callable] = None,
        device: str = "cuda",
        cache_size: int = 100,
        preload: bool = False
    ):
        """
        Initialize the PyTorchDataset.

        Args:
            data_paths: List of (structure_path, colors_path) tuples
            encoder_decoder: Encoder/decoder instance for preprocessing
            transform: Optional transform function to apply to samples
            device: Device to load tensors to
            cache_size: Number of samples to keep in memory cache
            preload: Whether to preload all data into memory
        """
        self.data_paths = data_paths
        self.encoder_decoder = encoder_decoder
        self.transform = transform
        self.device = device
        self.cache_size = cache_size
        self.preload = preload
        
        # Initialize cache
        self.cache = {}
        self.cache_order = []
        
        # Preload data if requested
        if self.preload:
            self._preload_all_data()

    def _preload_all_data(self):
        """Preload all data into memory."""
        log_info(f"Preloading {len(self.data_paths)} samples into memory...")
        for idx in tqdm(range(len(self.data_paths)), desc="Preloading"):
            self._load_sample(idx)

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample from disk.

        Args:
            idx: Sample index

        Returns:
            Tuple of (structure_tensor, colors_tensor)
        """
        if idx in self.cache:
            return self.cache[idx]

        structure_path, colors_path = self.data_paths[idx]
        
        try:
            # Load numpy arrays
            structure = np.load(structure_path, allow_pickle=True)
            colors = np.load(colors_path, allow_pickle=True)
            
            # Convert to tensors
            structure_tensor = torch.from_numpy(structure).to(self.device)
            colors_tensor = torch.from_numpy(colors).to(self.device)
            
            # Cache if we have space
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (structure_tensor, colors_tensor)
                self.cache_order.append(idx)
            elif not self.preload:
                # Remove oldest cached item
                oldest_idx = self.cache_order.pop(0)
                del self.cache[oldest_idx]
                self.cache[idx] = (structure_tensor, colors_tensor)
                self.cache_order.append(idx)
            
            return structure_tensor, colors_tensor
            
        except Exception as e:
            log_error(f"Error loading sample {idx}: {str(e)}")
            # Return empty tensors as fallback
            return (
                torch.zeros((64, 64, 64), device=self.device),
                torch.zeros((64, 64, 64), device=self.device)
            )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data
        """
        structure, colors = self._load_sample(idx)
        
        sample = {
            'structure': structure,
            'colors': colors,
            'index': torch.tensor(idx, device=self.device)
        }
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class PyTorchCurator:
    """
    PyTorch-based class for preprocessing sculpture data for machine learning.
    """

    def __init__(
        self,
        processing_method: str = "OHE",
        output_dir: str = "processed_data",
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        sparse_threshold: float = 0.1,
        verbose: bool = False,
    ):
        """
        Initialize the PyTorchCurator instance.

        Args:
            processing_method: Type of encoding to use ('OHE', 'BINARY', 'RGB', 'EMBEDDING')
            output_dir: Directory to save processed data
            device: Device to use for computations
            batch_size: Batch size for processing
            num_workers: Number of worker processes for data loading
            sparse_threshold: Threshold for sparse tensor conversion
            verbose: Whether to print detailed information
        """
        self.processing_method = processing_method.upper()
        self.output_dir = output_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sparse_threshold = sparse_threshold
        self.verbose = verbose
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize encoder/decoder
        self.encoder_decoder = None

    def load_samples_from_collection(
        self,
        collection_dir: str,
        limit: Optional[int] = None,
        shuffle: bool = True
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        """
        Load sample paths from a collection directory.

        Args:
            collection_dir: Path to the collection (date) directory
            limit: Maximum number of samples to load (None for all)
            shuffle: Whether to shuffle the samples

        Returns:
            Tuple of (data_paths, metadata)
        """
        begin_section(
            f"Loading sample paths from collection {os.path.basename(collection_dir)}"
        )

        try:
            # Path to samples directory
            samples_dir = os.path.join(collection_dir, "samples")

            if not os.path.exists(samples_dir):
                raise ValueError(f"Samples directory not found: {samples_dir}")

            # Check directory structure
            structures_dir = os.path.join(samples_dir, "structures")
            colors_dir = os.path.join(samples_dir, "colors")

            structure_files = []
            colors_files = []

            # Try new directory structure first
            if os.path.exists(structures_dir) and os.path.exists(colors_dir):
                structure_files.extend(glob.glob(os.path.join(structures_dir, "structure_*.npy")))
                structure_files.extend(glob.glob(os.path.join(structures_dir, "volume_*.npy")))
                colors_files.extend(glob.glob(os.path.join(colors_dir, "colors_*.npy")))
                colors_files.extend(glob.glob(os.path.join(colors_dir, "material_*.npy")))

            # Fallback to main samples directory
            if not structure_files:
                structure_files.extend(glob.glob(os.path.join(samples_dir, "volume_*.npy")))
                structure_files.extend(glob.glob(os.path.join(samples_dir, "structure_*.npy")))
                colors_files.extend(glob.glob(os.path.join(samples_dir, "material_*.npy")))
                colors_files.extend(glob.glob(os.path.join(samples_dir, "colors_*.npy")))

            log_info(f"Found {len(structure_files)} structure files and {len(colors_files)} color files")

            if not structure_files or not colors_files:
                raise ValueError(f"No matching structure/color files found in {samples_dir}")

            # Sort files
            structure_files = sorted(structure_files)
            colors_files = sorted(colors_files)

            # Match files based on sample numbers
            paired_files = []
            color_file_map = {}
            
            for color_file in colors_files:
                basename = os.path.basename(color_file)
                if "colors_" in basename:
                    sample_num = basename.replace("colors_", "").replace(".npy", "")
                elif "material_" in basename:
                    sample_num = basename.replace("material_", "").replace(".npy", "")
                else:
                    continue
                color_file_map[sample_num] = color_file

            for struct_file in structure_files:
                basename = os.path.basename(struct_file)
                if "structure_" in basename:
                    sample_num = basename.replace("structure_", "").replace(".npy", "")
                elif "volume_" in basename:
                    sample_num = basename.replace("volume_", "").replace(".npy", "")
                else:
                    continue

                if sample_num in color_file_map:
                    paired_files.append((struct_file, color_file_map[sample_num]))

            if not paired_files and len(structure_files) == len(colors_files):
                log_info("Using index-based pairing since file counts match")
                paired_files = list(zip(structure_files, colors_files))

            if not paired_files:
                raise ValueError(f"Could not match structure and color files in {samples_dir}")

            log_info(f"Successfully paired {len(paired_files)} files")

            # Shuffle if requested
            if shuffle:
                random.shuffle(paired_files)

            # Apply limit if specified
            if limit is not None and limit > 0:
                paired_files = paired_files[:limit]

            metadata = {
                'collection_dir': collection_dir,
                'total_samples': len(paired_files),
                'processing_method': self.processing_method,
                'device': self.device
            }

            log_success(f"Loaded {len(paired_files)} sample paths")
            end_section()

            return paired_files, metadata

        except Exception as e:
            log_error(f"Error loading sample paths: {str(e)}")
            end_section("Sample path loading failed")
            raise

    def create_dataset(
        self,
        data_paths: List[Tuple[str, str]],
        transform: Optional[Callable] = None,
        cache_size: int = 100,
        preload: bool = False
    ) -> PyTorchDataset:
        """
        Create a PyTorch dataset from data paths.

        Args:
            data_paths: List of (structure_path, colors_path) tuples
            transform: Optional transform function
            cache_size: Number of samples to cache in memory
            preload: Whether to preload all data

        Returns:
            PyTorchDataset instance
        """
        begin_section("Creating PyTorch Dataset")

        try:
            # Load a sample to initialize encoder/decoder
            if not self.encoder_decoder and data_paths:
                sample_colors = np.load(data_paths[0][1], allow_pickle=True)
                
                if self.processing_method == "OHE":
                    self.encoder_decoder = PyTorchOneHotEncoderDecoder(
                        sample_colors, device=self.device, verbose=self.verbose
                    )
                elif self.processing_method == "BINARY":
                    self.encoder_decoder = PyTorchBinaryEncoderDecoder(
                        sample_colors, device=self.device, verbose=self.verbose
                    )
                elif self.processing_method == "RGB":
                    self.encoder_decoder = PyTorchRGBEncoderDecoder(
                        sample_colors, device=self.device, verbose=self.verbose
                    )
                elif self.processing_method == "EMBEDDING":
                    self.encoder_decoder = PyTorchEmbeddingEncoderDecoder(
                        sample_colors, device=self.device, verbose=self.verbose
                    )
                else:
                    raise ValueError(f"Unknown processing method: {self.processing_method}")

            dataset = PyTorchDataset(
                data_paths=data_paths,
                encoder_decoder=self.encoder_decoder,
                transform=transform,
                device=self.device,
                cache_size=cache_size,
                preload=preload
            )

            log_success(f"Created dataset with {len(dataset)} samples")
            end_section()

            return dataset

        except Exception as e:
            log_error(f"Error creating dataset: {str(e)}")
            end_section("Dataset creation failed")
            raise

    def create_dataloader(
        self,
        dataset: PyTorchDataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset.

        Args:
            dataset: PyTorchDataset instance
            batch_size: Batch size (uses instance default if None)
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch

        Returns:
            DataLoader instance
        """
        begin_section("Creating PyTorch DataLoader")

        try:
            if batch_size is None:
                batch_size = self.batch_size

            # Custom collate function to handle variable-sized data
            def collate_fn(batch):
                """Custom collate function for batching samples."""
                structures = torch.stack([item['structure'] for item in batch])
                colors = torch.stack([item['colors'] for item in batch])
                indices = torch.stack([item['index'] for item in batch])
                
                return {
                    'structures': structures,
                    'colors': colors,
                    'indices': indices
                }

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=drop_last,
                collate_fn=collate_fn,
                pin_memory=True if self.device == "cuda" else False
            )

            log_success(f"Created DataLoader with batch_size={batch_size}")
            end_section()

            return dataloader

        except Exception as e:
            log_error(f"Error creating DataLoader: {str(e)}")
            end_section("DataLoader creation failed")
            raise

    def preprocess_batch(
        self,
        batch: Dict[str, torch.Tensor],
        apply_encoding: bool = True,
        apply_augmentation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of data.

        Args:
            batch: Batch dictionary containing structures and colors
            apply_encoding: Whether to apply encoding
            apply_augmentation: Whether to apply data augmentation

        Returns:
            Preprocessed batch dictionary
        """
        try:
            structures = batch['structures']
            colors = batch['colors']
            
            # Apply data augmentation if requested
            if apply_augmentation:
                structures, colors = self._apply_augmentation(structures, colors)
            
            # Apply encoding if requested
            if apply_encoding and self.encoder_decoder:
                if self.processing_method == "OHE":
                    encoded_colors, _ = self.encoder_decoder.ohe_encode()
                elif self.processing_method == "BINARY":
                    encoded_colors, _ = self.encoder_decoder.binary_encode()
                elif self.processing_method == "RGB":
                    encoded_colors, _ = self.encoder_decoder.rgb_encode()
                elif self.processing_method == "EMBEDDING":
                    encoded_colors, _ = self.encoder_decoder.embedding_encode()
                else:
                    encoded_colors = colors
                
                batch['encoded_colors'] = encoded_colors
            
            # Convert to sparse tensors if beneficial
            if self._should_use_sparse(structures):
                batch['structures'] = structures.to_sparse()
            
            batch['structures'] = structures
            batch['colors'] = colors
            
            return batch
            
        except Exception as e:
            log_error(f"Error preprocessing batch: {str(e)}")
            raise

    def _apply_augmentation(
        self,
        structures: torch.Tensor,
        colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation to structures and colors.

        Args:
            structures: Structure tensors
            colors: Color tensors

        Returns:
            Augmented (structures, colors) tensors
        """
        # Random rotation (90-degree increments)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            structures = torch.rot90(structures, k=k, dims=(-2, -1))
            colors = torch.rot90(colors, k=k, dims=(-2, -1))
        
        # Random flip
        if random.random() < 0.5:
            dim = random.choice([-3, -2, -1])
            structures = torch.flip(structures, dims=[dim])
            colors = torch.flip(colors, dims=[dim])
        
        return structures, colors

    def _should_use_sparse(self, tensor: torch.Tensor) -> bool:
        """
        Determine if a tensor should be converted to sparse format.

        Args:
            tensor: Input tensor

        Returns:
            True if tensor should be sparse
        """
        sparsity = 1.0 - (torch.count_nonzero(tensor).float() / tensor.numel())
        return sparsity > self.sparse_threshold

    def preprocess_collection(
        self,
        collection_dir: str,
        limit: Optional[int] = None,
        save_processed: bool = True,
        output_format: str = "pytorch"
    ) -> Dict[str, Any]:
        """
        Preprocess an entire collection with efficient batch processing.

        Args:
            collection_dir: Path to collection directory
            limit: Maximum number of samples to process
            save_processed: Whether to save processed data
            output_format: Output format ('pytorch', 'numpy', 'hdf5')

        Returns:
            Dictionary containing processing results and metadata
        """
        begin_section(f"Preprocessing collection {os.path.basename(collection_dir)}")

        try:
            # Load sample paths
            data_paths, metadata = self.load_samples_from_collection(
                collection_dir, limit=limit, shuffle=False
            )

            # Create dataset and dataloader
            dataset = self.create_dataset(data_paths, cache_size=min(100, len(data_paths)))
            dataloader = self.create_dataloader(dataset, shuffle=False)

            # Process batches
            processed_batches = []
            total_samples = 0

            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                # Preprocess batch
                processed_batch = self.preprocess_batch(
                    batch, apply_encoding=True, apply_augmentation=False
                )
                
                processed_batches.append(processed_batch)
                total_samples += len(batch['structures'])

                if self.verbose and (batch_idx + 1) % 10 == 0:
                    log_info(f"Processed {batch_idx + 1} batches, {total_samples} samples")

            # Save processed data if requested
            if save_processed:
                output_path = self._save_processed_data(
                    processed_batches, metadata, output_format
                )
                metadata['output_path'] = output_path

            metadata.update({
                'total_processed': total_samples,
                'num_batches': len(processed_batches),
                'processing_time': time.time(),
                'output_format': output_format
            })

            log_success(f"Successfully preprocessed {total_samples} samples")
            end_section()

            return {
                'processed_batches': processed_batches,
                'metadata': metadata
            }

        except Exception as e:
            log_error(f"Error preprocessing collection: {str(e)}")
            end_section("Collection preprocessing failed")
            raise

    def _save_processed_data(
        self,
        processed_batches: List[Dict[str, torch.Tensor]],
        metadata: Dict[str, Any],
        output_format: str
    ) -> str:
        """
        Save processed data to disk.

        Args:
            processed_batches: List of processed batch dictionaries
            metadata: Processing metadata
            output_format: Output format

        Returns:
            Path to saved data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.output_dir,
            f"processed_{self.processing_method.lower()}_{timestamp}"
        )
        os.makedirs(output_path, exist_ok=True)

        if output_format == "pytorch":
            # Save as PyTorch tensors
            for batch_idx, batch in enumerate(processed_batches):
                batch_path = os.path.join(output_path, f"batch_{batch_idx:04d}.pt")
                torch.save(batch, batch_path)
        
        elif output_format == "numpy":
            # Save as NumPy arrays
            for batch_idx, batch in enumerate(processed_batches):
                batch_dir = os.path.join(output_path, f"batch_{batch_idx:04d}")
                os.makedirs(batch_dir, exist_ok=True)
                
                for key, tensor in batch.items():
                    if isinstance(tensor, torch.Tensor):
                        np.save(
                            os.path.join(batch_dir, f"{key}.npy"),
                            tensor.cpu().numpy()
                        )
        
        elif output_format == "hdf5":
            # Save as HDF5 (requires h5py)
            try:
                import h5py
                hdf5_path = os.path.join(output_path, "processed_data.h5")
                
                with h5py.File(hdf5_path, 'w') as f:
                    for batch_idx, batch in enumerate(processed_batches):
                        batch_group = f.create_group(f"batch_{batch_idx:04d}")
                        for key, tensor in batch.items():
                            if isinstance(tensor, torch.Tensor):
                                batch_group.create_dataset(
                                    key, data=tensor.cpu().numpy()
                                )
            except ImportError:
                log_warning("h5py not available, falling back to PyTorch format")
                return self._save_processed_data(processed_batches, metadata, "pytorch")

        # Save metadata
        metadata_path = os.path.join(output_path, "metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            # Convert non-serializable items
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_metadata[key] = value
                else:
                    serializable_metadata[key] = str(value)
            json.dump(serializable_metadata, f, indent=2)

        log_info(f"Saved processed data to {output_path}")
        return output_path

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        if torch.cuda.is_available():
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                'cpu_memory': process.memory_info().rss / 1024**3,  # GB
            }

    def optimize_batch_size(
        self,
        dataset: PyTorchDataset,
        target_memory_gb: float = 8.0,
        max_batch_size: int = 128
    ) -> int:
        """
        Automatically determine optimal batch size based on memory constraints.

        Args:
            dataset: Dataset to test with
            target_memory_gb: Target memory usage in GB
            max_batch_size: Maximum batch size to test

        Returns:
            Optimal batch size
        """
        begin_section("Optimizing batch size")

        try:
            optimal_batch_size = 1
            
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
                if batch_size > max_batch_size:
                    break
                    
                try:
                    # Create test dataloader
                    test_loader = self.create_dataloader(
                        dataset, batch_size=batch_size, shuffle=False
                    )
                    
                    # Process one batch to measure memory
                    batch = next(iter(test_loader))
                    processed_batch = self.preprocess_batch(batch)
                    
                    # Check memory usage
                    memory_usage = self.get_memory_usage()
                    current_memory = (
                        memory_usage.get('gpu_allocated', 0) or 
                        memory_usage.get('cpu_memory', 0)
                    )
                    
                    if current_memory <= target_memory_gb:
                        optimal_batch_size = batch_size
                    else:
                        break
                        
                    # Clean up
                    del batch, processed_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    log_warning(f"Error testing batch_size {batch_size}: {str(e)}")
                    break

            log_success(f"Optimal batch size: {optimal_batch_size}")
            end_section()
            
            return optimal_batch_size

        except Exception as e:
            log_error(f"Error optimizing batch size: {str(e)}")
            end_section("Batch size optimization failed")
            return 1