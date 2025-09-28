"""
Dataset streaming components for DeepSculpt PyTorch implementation.

This module provides efficient streaming and loading of large 3D datasets
with memory optimization, caching, and distributed loading capabilities.
"""

import os
import time
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import warnings

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm

# Optional dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

from .data_generator import DataGenerator


class StreamingDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for PyTorch training.
    
    Generates samples on-demand without storing entire dataset in memory,
    with optional caching and prefetching for improved performance.
    """
    
    def __init__(
        self,
        data_generator: DataGenerator,
        num_samples: int,
        cache_size: int = 100,
        prefetch_size: int = 10,
        seed: Optional[int] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_generator: Data generator instance
            num_samples: Total number of samples in dataset
            cache_size: Number of samples to cache in memory
            prefetch_size: Number of samples to prefetch
            seed: Random seed for reproducibility
            transform: Optional transform to apply to samples
        """
        super().__init__()
        self.data_generator = data_generator
        self.num_samples = num_samples
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.seed = seed
        self.transform = transform
        
        # Cache for recently generated samples
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_order: List[int] = []
        self._cache_lock = threading.Lock()
        
        # Prefetch queue
        self._prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        # Start prefetch thread
        self._start_prefetch()
        
        try:
            for idx in range(self.num_samples):
                sample = self._get_sample(idx)
                
                if self.transform:
                    sample = self.transform(sample)
                
                yield sample
        finally:
            # Stop prefetch thread
            self._stop_prefetch_thread()
    
    def _get_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        # Check cache first
        with self._cache_lock:
            if idx in self._cache:
                structure, colors = self._cache[idx]
                return {
                    "structure": structure.clone(),
                    "colors": colors.clone(),
                    "index": torch.tensor(idx)
                }
        
        # Try to get from prefetch queue
        try:
            sample_data = self._prefetch_queue.get_nowait()
            structure, colors = sample_data
        except queue.Empty:
            # Generate on-demand
            structure, colors = self.data_generator.generate_single_sample()
        
        # Add to cache
        self._add_to_cache(idx, structure, colors)
        
        return {
            "structure": structure,
            "colors": colors,
            "index": torch.tensor(idx)
        }
    
    def _add_to_cache(self, idx: int, structure: torch.Tensor, colors: torch.Tensor):
        """Add sample to cache."""
        with self._cache_lock:
            if len(self._cache) >= self.cache_size and self.cache_size > 0:
                # Remove oldest item from cache
                oldest_idx = self._cache_order.pop(0)
                del self._cache[oldest_idx]
            
            if self.cache_size > 0:
                self._cache[idx] = (structure.clone(), colors.clone())
                self._cache_order.append(idx)
    
    def _start_prefetch(self):
        """Start prefetch thread."""
        if self.prefetch_size > 0 and self._prefetch_thread is None:
            self._stop_prefetch.clear()
            self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self._prefetch_thread.daemon = True
            self._prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Worker function for prefetching samples."""
        while not self._stop_prefetch.is_set():
            try:
                if not self._prefetch_queue.full():
                    structure, colors = self.data_generator.generate_single_sample()
                    self._prefetch_queue.put((structure, colors), timeout=1.0)
                else:
                    time.sleep(0.1)
            except queue.Full:
                continue
            except Exception as e:
                warnings.warn(f"Error in prefetch worker: {e}")
                break
    
    def _stop_prefetch_thread(self):
        """Stop prefetch thread."""
        if self._prefetch_thread is not None:
            self._stop_prefetch.set()
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None
    
    def clear_cache(self):
        """Clear the sample cache to free memory."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()


class FileBasedDataset(Dataset):
    """
    Dataset that loads samples from files on disk.
    
    Supports various file formats and provides efficient loading
    with optional caching and memory mapping.
    """
    
    def __init__(
        self,
        data_paths: List[str],
        file_format: str = "pytorch",
        cache_size: int = 0,
        memory_map: bool = False,
        transform: Optional[callable] = None,
        preload_metadata: bool = True
    ):
        """
        Initialize file-based dataset.
        
        Args:
            data_paths: List of paths to data files
            file_format: Format of data files ("pytorch", "numpy", "hdf5", "zarr")
            cache_size: Number of samples to cache in memory (0 = no caching)
            memory_map: Whether to use memory mapping for large files
            transform: Optional transform to apply to samples
            preload_metadata: Whether to preload file metadata
        """
        super().__init__()
        self.data_paths = data_paths
        self.file_format = file_format
        self.cache_size = cache_size
        self.memory_map = memory_map
        self.transform = transform
        
        # Validate file format
        self._validate_file_format()
        
        # Cache for loaded samples
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cache_order: List[int] = []
        
        # Metadata
        self.metadata = {}
        if preload_metadata:
            self._preload_metadata()
    
    def _validate_file_format(self):
        """Validate file format and availability of dependencies."""
        if self.file_format == "hdf5" and not HDF5_AVAILABLE:
            raise ImportError("h5py not available for HDF5 format")
        if self.file_format == "zarr" and not ZARR_AVAILABLE:
            raise ImportError("zarr not available for Zarr format")
    
    def _preload_metadata(self):
        """Preload metadata from files."""
        for i, path in enumerate(self.data_paths):
            try:
                if self.file_format == "pytorch":
                    # For PyTorch files, we can't easily get metadata without loading
                    self.metadata[i] = {"path": path, "size": os.path.getsize(path)}
                elif self.file_format == "numpy":
                    # For NumPy files, we can get shape info
                    with np.load(path, mmap_mode='r' if self.memory_map else None) as data:
                        self.metadata[i] = {
                            "path": path,
                            "shape": data.shape if hasattr(data, 'shape') else None,
                            "dtype": str(data.dtype) if hasattr(data, 'dtype') else None
                        }
                elif self.file_format == "hdf5":
                    with h5py.File(path, 'r') as f:
                        self.metadata[i] = {
                            "path": path,
                            "keys": list(f.keys()),
                            "shapes": {key: f[key].shape for key in f.keys()}
                        }
                elif self.file_format == "zarr":
                    root = zarr.open(path, mode='r')
                    self.metadata[i] = {
                        "path": path,
                        "keys": list(root.keys()),
                        "shapes": {key: root[key].shape for key in root.keys()}
                    }
            except Exception as e:
                warnings.warn(f"Could not load metadata for {path}: {e}")
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        # Check cache first
        if idx in self._cache:
            sample = self._cache[idx]
            if self.transform:
                sample = self.transform(sample)
            return sample
        
        # Load from file
        sample = self._load_sample(idx)
        
        # Add to cache
        self._add_to_cache(idx, sample)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sample from file."""
        path = self.data_paths[idx]
        
        try:
            if self.file_format == "pytorch":
                data = torch.load(path, map_location='cpu')
                if isinstance(data, dict):
                    return data
                else:
                    # Assume it's a tensor, wrap in dict
                    return {"data": data, "index": torch.tensor(idx)}
            
            elif self.file_format == "numpy":
                data = np.load(path, mmap_mode='r' if self.memory_map else None)
                if isinstance(data, np.ndarray):
                    return {
                        "data": torch.from_numpy(data.copy() if self.memory_map else data),
                        "index": torch.tensor(idx)
                    }
                else:
                    # Multiple arrays in file
                    return {
                        key: torch.from_numpy(arr.copy() if self.memory_map else arr)
                        for key, arr in data.items()
                    }
            
            elif self.file_format == "hdf5":
                with h5py.File(path, 'r') as f:
                    sample = {}
                    for key in f.keys():
                        data = f[key][:]
                        sample[key] = torch.from_numpy(data)
                    sample["index"] = torch.tensor(idx)
                    return sample
            
            elif self.file_format == "zarr":
                root = zarr.open(path, mode='r')
                sample = {}
                for key in root.keys():
                    data = root[key][:]
                    sample[key] = torch.from_numpy(data)
                sample["index"] = torch.tensor(idx)
                return sample
            
            else:
                raise ValueError(f"Unknown file format: {self.file_format}")
        
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} from {path}: {e}")
    
    def _add_to_cache(self, idx: int, sample: Dict[str, torch.Tensor]):
        """Add sample to cache."""
        if self.cache_size <= 0:
            return
        
        if len(self._cache) >= self.cache_size:
            # Remove oldest item from cache
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
        
        # Deep copy tensors for cache
        cached_sample = {key: tensor.clone() for key, tensor in sample.items()}
        self._cache[idx] = cached_sample
        self._cache_order.append(idx)
    
    def clear_cache(self):
        """Clear the sample cache to free memory."""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_metadata(self, idx: Optional[int] = None) -> Dict[str, Any]:
        """Get metadata for a specific sample or all samples."""
        if idx is not None:
            return self.metadata.get(idx, {})
        return self.metadata


class DistributedDataStreamer:
    """
    Distributed data streaming for multi-GPU training.
    
    Coordinates data generation and streaming across multiple processes
    with load balancing and fault tolerance.
    """
    
    def __init__(
        self,
        data_generator: DataGenerator,
        num_workers: int = 4,
        buffer_size: int = 100,
        timeout: float = 30.0
    ):
        """
        Initialize distributed data streamer.
        
        Args:
            data_generator: Data generator instance
            num_workers: Number of worker processes
            buffer_size: Size of data buffer
            timeout: Timeout for operations
        """
        self.data_generator = data_generator
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.timeout = timeout
        
        # Multiprocessing components
        self.manager = mp.Manager()
        self.data_queue = self.manager.Queue(maxsize=buffer_size)
        self.workers = []
        self.stop_event = self.manager.Event()
        
        # Statistics
        self.stats = self.manager.dict({
            "samples_generated": 0,
            "worker_errors": 0,
            "queue_full_count": 0
        })
    
    def start(self):
        """Start worker processes."""
        self.stop_event.clear()
        
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_process,
                args=(i, self.data_generator, self.data_queue, self.stop_event, self.stats)
            )
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop worker processes."""
        self.stop_event.set()
        
        for worker in self.workers:
            worker.join(timeout=self.timeout)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        self.workers.clear()
    
    def get_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Get a batch of samples from the stream.
        
        Args:
            batch_size: Number of samples in batch
            timeout: Timeout for getting samples
            
        Returns:
            List of sample dictionaries
        """
        timeout = timeout or self.timeout
        batch = []
        
        for _ in range(batch_size):
            try:
                sample = self.data_queue.get(timeout=timeout)
                batch.append(sample)
            except queue.Empty:
                warnings.warn(f"Timeout getting sample from queue, got {len(batch)}/{batch_size} samples")
                break
        
        return batch
    
    @staticmethod
    def _worker_process(
        worker_id: int,
        data_generator: DataGenerator,
        data_queue: mp.Queue,
        stop_event: mp.Event,
        stats: Dict[str, Any]
    ):
        """Worker process for generating data."""
        try:
            while not stop_event.is_set():
                try:
                    # Generate sample
                    structure, colors = data_generator.generate_single_sample()
                    
                    sample = {
                        "structure": structure,
                        "colors": colors,
                        "worker_id": torch.tensor(worker_id)
                    }
                    
                    # Put in queue
                    data_queue.put(sample, timeout=1.0)
                    stats["samples_generated"] += 1
                    
                except queue.Full:
                    stats["queue_full_count"] += 1
                    time.sleep(0.1)
                except Exception as e:
                    stats["worker_errors"] += 1
                    warnings.warn(f"Worker {worker_id} error: {e}")
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return dict(self.stats)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class DatasetSplitter:
    """
    Utility for splitting datasets into train/validation/test sets.
    """
    
    @staticmethod
    def split_file_paths(
        file_paths: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split file paths into train/val/test sets.
        
        Args:
            file_paths: List of file paths
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            shuffle: Whether to shuffle before splitting
            seed: Random seed for shuffling
            
        Returns:
            Tuple of (train_paths, val_paths, test_paths)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Shuffle if requested
        paths = file_paths.copy()
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(paths)
        
        # Calculate split indices
        n_total = len(paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]
        
        return train_paths, val_paths, test_paths
    
    @staticmethod
    def create_split_datasets(
        file_paths: List[str],
        file_format: str = "pytorch",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        **dataset_kwargs
    ) -> Tuple[FileBasedDataset, FileBasedDataset, FileBasedDataset]:
        """
        Create train/val/test datasets from file paths.
        
        Args:
            file_paths: List of file paths
            file_format: Format of data files
            split_ratios: (train_ratio, val_ratio, test_ratio)
            **dataset_kwargs: Additional arguments for FileBasedDataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_paths, val_paths, test_paths = DatasetSplitter.split_file_paths(
            file_paths, *split_ratios
        )
        
        train_dataset = FileBasedDataset(train_paths, file_format, **dataset_kwargs)
        val_dataset = FileBasedDataset(val_paths, file_format, **dataset_kwargs)
        test_dataset = FileBasedDataset(test_paths, file_format, **dataset_kwargs)
        
        return train_dataset, val_dataset, test_dataset


# Convenience functions
def create_streaming_dataloader(
    data_generator: DataGenerator,
    num_samples: int,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a streaming data loader."""
    dataset = StreamingDataset(data_generator, num_samples, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            key: torch.stack([sample[key] for sample in batch])
            for key in batch[0].keys()
        }
    )


def create_file_dataloader(
    file_paths: List[str],
    file_format: str = "pytorch",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a file-based data loader."""
    dataset = FileBasedDataset(file_paths, file_format, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )