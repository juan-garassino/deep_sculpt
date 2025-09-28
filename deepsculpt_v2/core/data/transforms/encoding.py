"""
Data encoding components for DeepSculpt PyTorch implementation.

This module provides various encoding schemes for 3D sculpture data including
one-hot encoding, binary encoding, RGB encoding, and learned embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.colors as mcolors
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from sklearn.preprocessing import LabelEncoder
import warnings


class BaseEncoder:
    """
    Base class for all encoders.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize base encoder.
        
        Args:
            device: Device for tensor operations
        """
        self.device = device
        self.is_fitted = False
    
    def fit(self, data: torch.Tensor):
        """
        Fit encoder to data.
        
        Args:
            data: Data to fit encoder on
        """
        self.is_fitted = True
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode data.
        
        Args:
            data: Data to encode
            
        Returns:
            Encoded data
        """
        raise NotImplementedError
    
    def decode(self, encoded_data: torch.Tensor) -> torch.Tensor:
        """
        Decode data.
        
        Args:
            encoded_data: Encoded data to decode
            
        Returns:
            Decoded data
        """
        raise NotImplementedError
    
    def to(self, device: str):
        """Move encoder to device."""
        self.device = device
        return self


class OneHotEncoder(BaseEncoder):
    """
    One-hot encoder for categorical data.
    
    Converts categorical values to one-hot encoded vectors.
    """
    
    def __init__(
        self,
        categories: Optional[List[Any]] = None,
        handle_unknown: str = "error",
        device: str = "cuda"
    ):
        """
        Initialize one-hot encoder.
        
        Args:
            categories: List of categories (auto-detected if None)
            handle_unknown: How to handle unknown categories ("error", "ignore")
            device: Device for tensor operations
        """
        super().__init__(device)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.category_to_idx = {}
        self.idx_to_category = {}
        self.n_categories = 0
    
    def fit(self, data: torch.Tensor):
        """Fit encoder to data."""
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        
        # Get unique categories
        if self.categories is None:
            unique_categories = np.unique(data_np.flatten())
            self.categories = sorted(unique_categories, key=lambda x: str(x))
        
        # Create mappings
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        self.n_categories = len(self.categories)
        
        super().fit(data)
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data to one-hot representation."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding")
        
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            original_shape = data.shape
        else:
            data_np = data
            original_shape = data_np.shape
        
        # Flatten data
        flat_data = data_np.flatten()
        
        # Convert categories to indices
        indices = []
        for item in flat_data:
            if item in self.category_to_idx:
                indices.append(self.category_to_idx[item])
            elif self.handle_unknown == "ignore":
                indices.append(0)  # Use first category as default
            else:
                raise ValueError(f"Unknown category: {item}")
        
        # Convert to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        # Create one-hot encoding
        one_hot = F.one_hot(indices_tensor, num_classes=self.n_categories).float()
        
        # Reshape to original shape + one-hot dimension
        new_shape = original_shape + (self.n_categories,)
        one_hot = one_hot.reshape(new_shape)
        
        return one_hot
    
    def decode(self, encoded_data: torch.Tensor) -> torch.Tensor:
        """Decode one-hot encoded data."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before decoding")
        
        # Get indices from one-hot encoding
        indices = torch.argmax(encoded_data, dim=-1)
        
        # Convert indices back to categories
        decoded_shape = indices.shape
        flat_indices = indices.flatten().cpu().numpy()
        
        decoded_flat = []
        for idx in flat_indices:
            if idx in self.idx_to_category:
                decoded_flat.append(self.idx_to_category[idx])
            else:
                decoded_flat.append(self.categories[0])  # Default to first category
        
        # Convert back to tensor
        decoded_array = np.array(decoded_flat).reshape(decoded_shape)
        return torch.from_numpy(decoded_array).to(self.device)
    
    def get_info(self) -> Dict[str, Any]:
        """Get encoder information."""
        return {
            "n_categories": self.n_categories,
            "categories": self.categories,
            "handle_unknown": self.handle_unknown,
            "is_fitted": self.is_fitted
        }


class BinaryEncoder(BaseEncoder):
    """
    Binary encoder for categorical data.
    
    Converts categorical values to binary representations.
    """
    
    def __init__(
        self,
        categories: Optional[List[Any]] = None,
        device: str = "cuda"
    ):
        """
        Initialize binary encoder.
        
        Args:
            categories: List of categories (auto-detected if None)
            device: Device for tensor operations
        """
        super().__init__(device)
        self.categories = categories
        self.label_encoder = LabelEncoder()
        self.n_bits = 0
    
    def fit(self, data: torch.Tensor):
        """Fit encoder to data."""
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        
        # Fit label encoder
        flat_data = data_np.flatten()
        self.label_encoder.fit(flat_data)
        
        # Get categories and calculate bits needed
        self.categories = self.label_encoder.classes_
        self.n_bits = int(np.ceil(np.log2(len(self.categories))))
        
        super().fit(data)
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data to binary representation."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding")
        
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            original_shape = data.shape
        else:
            data_np = data
            original_shape = data_np.shape
        
        # Flatten and encode
        flat_data = data_np.flatten()
        label_encoded = self.label_encoder.transform(flat_data)
        
        # Convert to PyTorch tensor
        label_tensor = torch.from_numpy(label_encoded).long().to(self.device)
        
        # Convert to binary representation
        binary_tensor = torch.zeros(
            len(label_encoded), self.n_bits, device=self.device, dtype=torch.float32
        )
        
        # Convert each label to binary using bit operations
        for bit in range(self.n_bits):
            binary_tensor[:, bit] = (label_tensor >> bit) & 1
        
        # Reshape to original shape + binary dimension
        new_shape = original_shape + (self.n_bits,)
        binary_tensor = binary_tensor.reshape(new_shape)
        
        return binary_tensor
    
    def decode(self, encoded_data: torch.Tensor) -> torch.Tensor:
        """Decode binary encoded data."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before decoding")
        
        # Reshape for decoding
        original_shape = encoded_data.shape[:-1]
        flat_encoded = encoded_data.reshape(-1, self.n_bits)
        
        # Convert binary vectors back to integers
        powers_of_2 = torch.pow(2, torch.arange(self.n_bits, device=self.device)).float()
        label_indices = torch.sum(flat_encoded * powers_of_2, dim=1).long()
        
        # Convert to CPU for sklearn inverse transform
        label_indices_cpu = label_indices.cpu().numpy()
        
        # Convert back to original categories
        decoded_categories = self.label_encoder.inverse_transform(label_indices_cpu)
        
        # Reshape and convert to tensor
        decoded_array = decoded_categories.reshape(original_shape)
        return torch.from_numpy(decoded_array).to(self.device)
    
    def get_info(self) -> Dict[str, Any]:
        """Get encoder information."""
        return {
            "n_categories": len(self.categories) if self.categories is not None else 0,
            "n_bits": self.n_bits,
            "categories": list(self.categories) if self.categories is not None else [],
            "is_fitted": self.is_fitted
        }


class RGBEncoder(BaseEncoder):
    """
    RGB encoder for color data.
    
    Converts color names/values to RGB representations.
    """
    
    def __init__(
        self,
        color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None,
        device: str = "cuda"
    ):
        """
        Initialize RGB encoder.
        
        Args:
            color_dict: Dictionary mapping colors to RGB values
            device: Device for tensor operations
        """
        super().__init__(device)
        self.color_dict = color_dict or self._create_default_color_dict()
    
    def _create_default_color_dict(self) -> Dict[Any, Tuple[int, int, int]]:
        """Create default color dictionary."""
        color_dict = {}
        
        # Add matplotlib colors
        for name, hex_color in mcolors.TABLEAU_COLORS.items():
            rgb = tuple(int(x * 255) for x in mcolors.to_rgb(hex_color))
            color_dict[name] = rgb
        
        for name, hex_color in mcolors.CSS4_COLORS.items():
            if name not in color_dict:
                rgb = tuple(int(x * 255) for x in mcolors.to_rgb(hex_color))
                color_dict[name] = rgb
        
        # Add special case for None
        color_dict[None] = (0, 0, 0)
        
        return color_dict
    
    def fit(self, data: torch.Tensor):
        """Fit encoder to data (no-op for RGB encoder)."""
        super().fit(data)
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data to RGB representation."""
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            original_shape = data.shape
        else:
            data_np = data
            original_shape = data_np.shape
        
        # Create RGB tensor
        rgb_shape = original_shape + (3,)
        rgb_tensor = torch.zeros(rgb_shape, dtype=torch.uint8, device=self.device)
        
        # Convert each color to RGB
        flat_data = data_np.flatten()
        flat_rgb = rgb_tensor.view(-1, 3)
        
        for i, color in enumerate(flat_data):
            if color in self.color_dict:
                flat_rgb[i] = torch.tensor(self.color_dict[color], device=self.device)
            else:
                # Default to gray for unknown colors
                flat_rgb[i] = torch.tensor((128, 128, 128), device=self.device)
        
        return rgb_tensor
    
    def decode(self, encoded_data: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
        """Decode RGB encoded data."""
        # Get original shape
        original_shape = encoded_data.shape[:-1]
        
        # Flatten RGB data
        flat_rgb = encoded_data.view(-1, 3).float()
        
        # Create reverse mapping
        color_list = list(self.color_dict.keys())
        rgb_list = [self.color_dict[color] for color in color_list]
        rgb_lookup = torch.tensor(rgb_list, dtype=torch.float32, device=self.device)
        
        # Find closest colors using Euclidean distance
        distances = torch.cdist(flat_rgb.unsqueeze(0), rgb_lookup.unsqueeze(0)).squeeze(0)
        min_distances, closest_indices = torch.min(distances, dim=1)
        
        # Apply threshold
        valid_matches = min_distances <= threshold
        
        # Create result array
        result = []
        for i, (valid, idx) in enumerate(zip(valid_matches, closest_indices)):
            if valid:
                result.append(color_list[idx])
            else:
                result.append(None)  # Unknown color
        
        # Reshape and convert to tensor
        result_array = np.array(result).reshape(original_shape)
        return torch.from_numpy(result_array).to(self.device)
    
    def get_info(self) -> Dict[str, Any]:
        """Get encoder information."""
        return {
            "n_colors": len(self.color_dict),
            "available_colors": list(self.color_dict.keys()),
            "is_fitted": self.is_fitted
        }


class LearnedEmbeddingEncoder(BaseEncoder):
    """
    Learned embedding encoder using neural networks.
    
    Learns embeddings for categorical data through training.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        categories: Optional[List[Any]] = None,
        device: str = "cuda"
    ):
        """
        Initialize learned embedding encoder.
        
        Args:
            embedding_dim: Dimension of learned embeddings
            categories: List of categories (auto-detected if None)
            device: Device for tensor operations
        """
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.categories = categories
        self.category_to_idx = {}
        self.n_categories = 0
        self.embedding = None
    
    def fit(self, data: torch.Tensor):
        """Fit encoder to data."""
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        
        # Get unique categories
        if self.categories is None:
            unique_categories = np.unique(data_np.flatten())
            self.categories = sorted(unique_categories, key=lambda x: str(x))
        
        # Create mappings
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.n_categories = len(self.categories)
        
        # Create embedding layer
        self.embedding = nn.Embedding(self.n_categories, self.embedding_dim).to(self.device)
        
        super().fit(data)
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data using learned embeddings."""
        if not self.is_fitted or self.embedding is None:
            raise ValueError("Encoder must be fitted before encoding")
        
        # Convert to numpy for processing
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            original_shape = data.shape
        else:
            data_np = data
            original_shape = data_np.shape
        
        # Convert categories to indices
        flat_data = data_np.flatten()
        indices = []
        for item in flat_data:
            if item in self.category_to_idx:
                indices.append(self.category_to_idx[item])
            else:
                indices.append(0)  # Default to first category
        
        # Convert to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        # Apply embedding
        embedded = self.embedding(indices_tensor)
        
        # Reshape to original shape + embedding dimension
        new_shape = original_shape + (self.embedding_dim,)
        embedded = embedded.reshape(new_shape)
        
        return embedded
    
    def decode(self, encoded_data: torch.Tensor) -> torch.Tensor:
        """Decode embedded data (approximate)."""
        if not self.is_fitted or self.embedding is None:
            raise ValueError("Encoder must be fitted before decoding")
        
        # Get original shape
        original_shape = encoded_data.shape[:-1]
        
        # Flatten embeddings
        flat_embeddings = encoded_data.view(-1, self.embedding_dim)
        
        # Get all embedding vectors
        all_indices = torch.arange(self.n_categories, device=self.device)
        all_embeddings = self.embedding(all_indices)
        
        # Find closest embeddings using cosine similarity
        similarities = F.cosine_similarity(
            flat_embeddings.unsqueeze(1),
            all_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Get indices of most similar embeddings
        closest_indices = torch.argmax(similarities, dim=1)
        
        # Convert back to categories
        result = []
        for idx in closest_indices.cpu().numpy():
            result.append(self.categories[idx])
        
        # Reshape and convert to tensor
        result_array = np.array(result).reshape(original_shape)
        return torch.from_numpy(result_array).to(self.device)
    
    def get_embedding_layer(self) -> nn.Embedding:
        """Get the embedding layer for training."""
        return self.embedding
    
    def get_info(self) -> Dict[str, Any]:
        """Get encoder information."""
        return {
            "n_categories": self.n_categories,
            "embedding_dim": self.embedding_dim,
            "categories": self.categories,
            "is_fitted": self.is_fitted,
            "trainable_parameters": sum(p.numel() for p in self.embedding.parameters()) if self.embedding else 0
        }


class CompositeEncoder:
    """
    Composite encoder that applies different encoders to different parts of data.
    """
    
    def __init__(
        self,
        encoder_map: Dict[str, BaseEncoder],
        device: str = "cuda"
    ):
        """
        Initialize composite encoder.
        
        Args:
            encoder_map: Mapping from data keys to encoders
            device: Device for tensor operations
        """
        self.encoder_map = encoder_map
        self.device = device
        self.is_fitted = False
    
    def fit(self, data: Dict[str, torch.Tensor]):
        """Fit all encoders to their respective data."""
        for key, encoder in self.encoder_map.items():
            if key in data:
                encoder.fit(data[key])
        self.is_fitted = True
    
    def encode(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode data using appropriate encoders."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding")
        
        result = {}
        for key, tensor in data.items():
            if key in self.encoder_map:
                result[key] = self.encoder_map[key].encode(tensor)
            else:
                result[key] = tensor
        
        return result
    
    def decode(self, encoded_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode data using appropriate encoders."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before decoding")
        
        result = {}
        for key, tensor in encoded_data.items():
            if key in self.encoder_map:
                result[key] = self.encoder_map[key].decode(tensor)
            else:
                result[key] = tensor
        
        return result
    
    def to(self, device: str):
        """Move all encoders to device."""
        self.device = device
        for encoder in self.encoder_map.values():
            encoder.to(device)
        return self


# Convenience functions
def create_standard_encoder(
    encoding_type: str = "one_hot",
    device: str = "cuda",
    **kwargs
) -> BaseEncoder:
    """Create a standard encoder."""
    if encoding_type == "one_hot":
        return OneHotEncoder(device=device, **kwargs)
    elif encoding_type == "binary":
        return BinaryEncoder(device=device, **kwargs)
    elif encoding_type == "rgb":
        return RGBEncoder(device=device, **kwargs)
    elif encoding_type == "embedding":
        return LearnedEmbeddingEncoder(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def create_sculpture_encoder(device: str = "cuda") -> CompositeEncoder:
    """Create an encoder optimized for sculpture data."""
    encoder_map = {
        "structure": BinaryEncoder(device=device),
        "colors": RGBEncoder(device=device)
    }
    return CompositeEncoder(encoder_map, device=device)