"""
Data loaders for DeepSculpt PyTorch implementation.

This module provides specialized data loaders for 3D sculpture data
with support for various formats, memory optimization, and distributed loading.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable, Iterator
import random
import warnings
from pathlib import Path

from ..generation.data_generator import DataGenerator
from ..generation.dataset_streamer import StreamingDataset, FileBasedDataset
from ..transforms.preprocessing import BasePreprocessor
from ..transforms.encoding import BaseEncoder


class AdaptiveBatchSampler(Sampler):
    """
    Adaptive batch sampler that adjusts batch size based on memory usage.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        initial_batch_size: int = 32,
        max_batch_size: int = 128,
        min_batch_size: int = 1,
        memory_threshold: float = 0.8,
        adaptation_factor: float = 0.9
    ):
        """
        Initialize adaptive batch sampler.
        
        Args:
            dataset: Dataset to sample from
            initial_batch_size: Initial batch size
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            memory_threshold: Memory usage threshold for adaptation
            adaptation_factor: Factor to reduce batch size when memory is high
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_threshold = memory_threshold
        self.adaptation_factor = adaptation_factor
        
        self.indices = list(range(len(dataset)))
        self.memory_history = []
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches with adaptive sizing."""
        random.shuffle(self.indices)
        
        i = 0
        while i < len(self.indices):
            # Check memory usage and adapt batch size
            self._adapt_batch_size()
            
            # Create batch
            batch_end = min(i + self.current_batch_size, len(self.indices))
            batch_indices = self.indices[i:batch_end]
            
            yield batch_indices
            i = batch_end
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self.dataset) + self.current_batch_size - 1) // self.current_batch_size
    
    def _adapt_batch_size(self):
        """Adapt batch size based on memory usage."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.memory_history.append(memory_used)
            
            # Keep only recent history
            if len(self.memory_history) > 10:
                self.memory_history = self.memory_history[-10:]
            
            # Adapt batch size
            avg_memory = np.mean(self.memory_history)
            if avg_memory > self.memory_threshold:
                # Reduce batch size
                new_batch_size = max(
                    self.min_batch_size,
                    int(self.current_batch_size * self.adaptation_factor)
                )
                if new_batch_size != self.current_batch_size:
                    self.current_batch_size = new_batch_size
                    warnings.warn(f"Reduced batch size to {self.current_batch_size} due to memory pressure")
            elif avg_memory < self.memory_threshold * 0.7:
                # Increase batch size
                new_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size / self.adaptation_factor)
                )
                if new_batch_size != self.current_batch_size:
                    self.current_batch_size = new_batch_size


class BalancedSampler(Sampler):
    """
    Balanced sampler for ensuring equal representation of different classes/conditions.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        label_key: str = "label",
        samples_per_class: Optional[int] = None
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            label_key: Key for labels in dataset samples
            samples_per_class: Number of samples per class (None for equal to smallest class)
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.label_key = label_key
        self.samples_per_class = samples_per_class
        
        # Build class indices
        self.class_indices = self._build_class_indices()
        self.classes = list(self.class_indices.keys())
        
        # Determine samples per class
        if self.samples_per_class is None:
            self.samples_per_class = min(len(indices) for indices in self.class_indices.values())
    
    def _build_class_indices(self) -> Dict[Any, List[int]]:
        """Build mapping from classes to sample indices."""
        class_indices = {}
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if isinstance(sample, dict) and self.label_key in sample:
                    label = sample[self.label_key]
                    if hasattr(label, 'item'):  # Handle tensor labels
                        label = label.item()
                    
                    if label not in class_indices:
                        class_indices[label] = []
                    class_indices[label].append(idx)
            except Exception as e:
                warnings.warn(f"Could not get label for sample {idx}: {e}")
        
        return class_indices
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over balanced samples."""
        # Sample from each class
        sampled_indices = []
        
        for class_label in self.classes:
            class_idx_list = self.class_indices[class_label]
            if len(class_idx_list) >= self.samples_per_class:
                # Sample without replacement
                sampled = random.sample(class_idx_list, self.samples_per_class)
            else:
                # Sample with replacement
                sampled = random.choices(class_idx_list, k=self.samples_per_class)
            
            sampled_indices.extend(sampled)
        
        # Shuffle the combined samples
        random.shuffle(sampled_indices)
        
        for idx in sampled_indices:
            yield idx
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self.classes) * self.samples_per_class


class MultiFormatDataLoader:
    """
    Data loader that can handle multiple data formats and sources.
    """
    
    def __init__(
        self,
        datasets: Union[Dataset, List[Dataset]],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        preprocessor: Optional[BasePreprocessor] = None,
        encoder: Optional[BaseEncoder] = None,
        collate_fn: Optional[Callable] = None,
        sampler: Optional[Sampler] = None,
        **dataloader_kwargs
    ):
        """
        Initialize multi-format data loader.
        
        Args:
            datasets: Single dataset or list of datasets
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            preprocessor: Optional preprocessor
            encoder: Optional encoder
            collate_fn: Custom collate function
            sampler: Custom sampler
            **dataloader_kwargs: Additional DataLoader arguments
        """
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.preprocessor = preprocessor
        self.encoder = encoder
        
        # Create combined dataset if multiple datasets
        if len(self.datasets) > 1:
            self.combined_dataset = CombinedDataset(self.datasets)
        else:
            self.combined_dataset = self.datasets[0]
        
        # Set up collate function
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        # Create data loader
        self.dataloader = DataLoader(
            self.combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=sampler,
            **dataloader_kwargs
        )
    
    def _default_collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Default collate function for batching samples."""
        # Filter out None samples (filtered by preprocessor)
        batch = [sample for sample in batch if sample is not None]
        
        if not batch:
            # Return empty batch
            return {}
        
        # Apply preprocessing
        if self.preprocessor:
            processed_batch = []
            for sample in batch:
                processed_sample = self.preprocessor(sample)
                if processed_sample is not None:
                    processed_batch.append(processed_sample)
            batch = processed_batch
        
        if not batch:
            return {}
        
        # Apply encoding
        if self.encoder:
            encoded_batch = []
            for sample in batch:
                if isinstance(sample, dict):
                    encoded_sample = self.encoder.encode(sample)
                    encoded_batch.append(encoded_sample)
                else:
                    encoded_batch.append(sample)
            batch = encoded_batch
        
        # Collate into batch tensors
        if not batch:
            return {}
        
        collated = {}
        for key in batch[0].keys():
            try:
                values = [sample[key] for sample in batch]
                if all(isinstance(v, torch.Tensor) for v in values):
                    # Stack tensors
                    collated[key] = torch.stack(values, dim=0)
                else:
                    # Keep as list for non-tensor data
                    collated[key] = values
            except Exception as e:
                warnings.warn(f"Could not collate key {key}: {e}")
                collated[key] = [sample[key] for sample in batch]
        
        return collated
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get number of batches."""
        return len(self.dataloader)


class CombinedDataset(Dataset):
    """
    Dataset that combines multiple datasets.
    """
    
    def __init__(self, datasets: List[Dataset]):
        """
        Initialize combined dataset.
        
        Args:
            datasets: List of datasets to combine
        """
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()
    
    def _get_cumulative_sizes(self) -> List[int]:
        """Get cumulative sizes for indexing."""
        cumulative_sizes = []
        cumsum = 0
        for dataset in self.datasets:
            cumsum += len(dataset)
            cumulative_sizes.append(cumsum)
        return cumulative_sizes
    
    def __len__(self) -> int:
        """Get total length."""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Any:
        """Get item by global index."""
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        # Find which dataset the index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break
        
        # Calculate local index within the dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]


class StreamingDataLoader:
    """
    Specialized data loader for streaming datasets with memory optimization.
    """
    
    def __init__(
        self,
        data_generator: DataGenerator,
        num_samples: int,
        batch_size: int = 32,
        cache_size: int = 100,
        prefetch_size: int = 10,
        num_workers: int = 0,  # Streaming datasets work better with single process
        preprocessor: Optional[BasePreprocessor] = None,
        encoder: Optional[BaseEncoder] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize streaming data loader.
        
        Args:
            data_generator: Data generator for creating samples
            num_samples: Total number of samples
            batch_size: Batch size
            cache_size: Size of sample cache
            prefetch_size: Size of prefetch buffer
            num_workers: Number of worker processes (0 recommended for streaming)
            preprocessor: Optional preprocessor
            encoder: Optional encoder
            seed: Random seed
        """
        self.data_generator = data_generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.encoder = encoder
        
        # Create streaming dataset
        self.dataset = StreamingDataset(
            data_generator=data_generator,
            num_samples=num_samples,
            cache_size=cache_size,
            prefetch_size=prefetch_size,
            seed=seed
        )
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for streaming data."""
        # Apply preprocessing and encoding similar to MultiFormatDataLoader
        if self.preprocessor:
            processed_batch = []
            for sample in batch:
                processed_sample = self.preprocessor(sample)
                if processed_sample is not None:
                    processed_batch.append(processed_sample)
            batch = processed_batch
        
        if not batch:
            return {}
        
        if self.encoder:
            encoded_batch = []
            for sample in batch:
                encoded_sample = self.encoder.encode(sample)
                encoded_batch.append(encoded_sample)
            batch = encoded_batch
        
        # Collate
        collated = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = values
        
        return collated
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get number of batches."""
        return len(self.dataloader)


# Convenience functions
def create_training_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    preprocessor: Optional[BasePreprocessor] = None,
    encoder: Optional[BaseEncoder] = None,
    use_adaptive_batching: bool = False,
    **kwargs
) -> MultiFormatDataLoader:
    """Create a data loader optimized for training."""
    sampler = None
    if use_adaptive_batching:
        sampler = AdaptiveBatchSampler(dataset, initial_batch_size=batch_size)
        shuffle = False  # Sampler handles shuffling
    
    return MultiFormatDataLoader(
        datasets=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        preprocessor=preprocessor,
        encoder=encoder,
        sampler=sampler,
        **kwargs
    )


def create_validation_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    preprocessor: Optional[BasePreprocessor] = None,
    encoder: Optional[BaseEncoder] = None,
    **kwargs
) -> MultiFormatDataLoader:
    """Create a data loader optimized for validation."""
    return MultiFormatDataLoader(
        datasets=dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        preprocessor=preprocessor,
        encoder=encoder,
        **kwargs
    )


def create_streaming_dataloader(
    data_generator: DataGenerator,
    num_samples: int,
    batch_size: int = 32,
    preprocessor: Optional[BasePreprocessor] = None,
    encoder: Optional[BaseEncoder] = None,
    **kwargs
) -> StreamingDataLoader:
    """Create a streaming data loader."""
    return StreamingDataLoader(
        data_generator=data_generator,
        num_samples=num_samples,
        batch_size=batch_size,
        preprocessor=preprocessor,
        encoder=encoder,
        **kwargs
    )


def create_balanced_dataloader(
    dataset: Dataset,
    label_key: str = "label",
    batch_size: int = 32,
    samples_per_class: Optional[int] = None,
    num_workers: int = 4,
    preprocessor: Optional[BasePreprocessor] = None,
    encoder: Optional[BaseEncoder] = None,
    **kwargs
) -> MultiFormatDataLoader:
    """Create a balanced data loader."""
    sampler = BalancedSampler(
        dataset=dataset,
        label_key=label_key,
        samples_per_class=samples_per_class
    )
    
    return MultiFormatDataLoader(
        datasets=dataset,
        batch_size=batch_size,
        shuffle=False,  # Sampler handles sampling
        num_workers=num_workers,
        preprocessor=preprocessor,
        encoder=encoder,
        sampler=sampler,
        **kwargs
    )