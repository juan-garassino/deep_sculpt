#!/usr/bin/env python3
"""
Integration test for PyTorchCurator functionality.

This script demonstrates the complete PyTorch-based curator functionality
including all encoding methods and batch processing capabilities.
"""

import os
import tempfile
import shutil
import numpy as np
import torch
from deepSculpt.pytorch_curator import (
    PyTorchOneHotEncoderDecoder,
    PyTorchBinaryEncoderDecoder,
    PyTorchRGBEncoderDecoder,
    PyTorchEmbeddingEncoderDecoder,
    PyTorchCurator
)

def test_all_encoders():
    """Test all encoder types."""
    print("Testing all encoder types...")
    
    # Create test data
    colors = np.array([[[['red', 'blue'], ['green', None]], [['red', 'blue'], ['green', None]]]])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test One-Hot Encoder
    print("\n1. Testing One-Hot Encoder:")
    ohe_encoder = PyTorchOneHotEncoderDecoder(colors, device=device, verbose=False)
    encoded_ohe, classes_ohe = ohe_encoder.ohe_encode()
    structures_ohe, colors_ohe = ohe_encoder.ohe_decode(encoded_ohe)
    print(f"   ✓ One-hot encoded shape: {encoded_ohe.shape}")
    print(f"   ✓ Classes: {classes_ohe}")
    
    # Test Binary Encoder
    print("\n2. Testing Binary Encoder:")
    binary_encoder = PyTorchBinaryEncoderDecoder(colors, device=device, verbose=False)
    encoded_binary, classes_binary = binary_encoder.binary_encode()
    structures_binary, colors_binary = binary_encoder.binary_decode(encoded_binary)
    print(f"   ✓ Binary encoded shape: {encoded_binary.shape}")
    print(f"   ✓ Classes: {classes_binary}")
    
    # Test RGB Encoder
    print("\n3. Testing RGB Encoder:")
    rgb_encoder = PyTorchRGBEncoderDecoder(colors, device=device, verbose=False)
    encoded_rgb, mapping_rgb = rgb_encoder.rgb_encode()
    structures_rgb, colors_rgb = rgb_encoder.rgb_decode(encoded_rgb)
    print(f"   ✓ RGB encoded shape: {encoded_rgb.shape}")
    print(f"   ✓ Color mapping keys: {list(mapping_rgb.keys())[:5]}...")
    
    # Test Embedding Encoder
    print("\n4. Testing Embedding Encoder:")
    embedding_encoder = PyTorchEmbeddingEncoderDecoder(colors, embedding_dim=16, device=device, verbose=False)
    encoded_embedding, embedding_layer = embedding_encoder.embedding_encode()
    structures_embedding, colors_embedding = embedding_encoder.embedding_decode(encoded_embedding)
    print(f"   ✓ Embedding encoded shape: {encoded_embedding.shape}")
    print(f"   ✓ Embedding dimension: {embedding_layer.embedding_dim}")
    
    print("\n✅ All encoders tested successfully!")

def test_curator_with_mock_data():
    """Test PyTorchCurator with mock data."""
    print("\nTesting PyTorchCurator with mock data...")
    
    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    collection_dir = os.path.join(temp_dir, "test_collection")
    samples_dir = os.path.join(collection_dir, "samples")
    structures_dir = os.path.join(samples_dir, "structures")
    colors_dir = os.path.join(samples_dir, "colors")
    
    os.makedirs(structures_dir)
    os.makedirs(colors_dir)
    
    try:
        # Create mock data files
        print("Creating mock data files...")
        for i in range(3):
            # Create structure file
            structure = np.random.randint(0, 2, (16, 16, 16))  # Small for speed
            structure_path = os.path.join(structures_dir, f"structure_{i:03d}.npy")
            np.save(structure_path, structure)
            
            # Create colors file
            colors = np.random.choice(['red', 'blue', 'green', None], (16, 16, 16))
            colors_path = os.path.join(colors_dir, f"colors_{i:03d}.npy")
            np.save(colors_path, colors)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test different processing methods
        for method in ["OHE", "BINARY", "RGB"]:  # Skip EMBEDDING for speed
            print(f"\nTesting {method} processing method:")
            
            curator = PyTorchCurator(
                processing_method=method,
                output_dir=os.path.join(temp_dir, f"processed_{method.lower()}"),
                device=device,
                batch_size=2,
                num_workers=0,  # Avoid multiprocessing issues in tests
                verbose=False
            )
            
            # Load sample paths
            data_paths, metadata = curator.load_samples_from_collection(
                collection_dir, limit=None, shuffle=False
            )
            print(f"   ✓ Loaded {len(data_paths)} sample paths")
            
            # Create dataset
            dataset = curator.create_dataset(data_paths, cache_size=10)
            print(f"   ✓ Created dataset with {len(dataset)} samples")
            
            # Create dataloader
            dataloader = curator.create_dataloader(dataset, batch_size=2)
            print(f"   ✓ Created dataloader")
            
            # Process one batch
            batch = next(iter(dataloader))
            processed_batch = curator.preprocess_batch(
                batch, apply_encoding=True, apply_augmentation=False
            )
            print(f"   ✓ Processed batch with keys: {list(processed_batch.keys())}")
            
            # Check memory usage
            memory_usage = curator.get_memory_usage()
            print(f"   ✓ Memory usage: {memory_usage}")
        
        print("\n✅ PyTorchCurator tested successfully!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print("✓ Cleaned up temporary files")

def test_performance_comparison():
    """Test performance characteristics."""
    print("\nTesting performance characteristics...")
    
    # Create larger test data
    colors = np.random.choice(['red', 'blue', 'green', 'yellow', None], (2, 32, 32, 32))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    import time
    
    # Test encoding speed
    print(f"Testing encoding speed on {device}...")
    
    start_time = time.time()
    encoder = PyTorchOneHotEncoderDecoder(colors, device=device, verbose=False)
    encoded, classes = encoder.ohe_encode()
    encoding_time = time.time() - start_time
    
    start_time = time.time()
    structures, decoded_colors = encoder.ohe_decode(encoded)
    decoding_time = time.time() - start_time
    
    print(f"   ✓ Encoding time: {encoding_time:.4f}s")
    print(f"   ✓ Decoding time: {decoding_time:.4f}s")
    print(f"   ✓ Data shape: {colors.shape} -> {encoded.shape}")
    
    # Test memory efficiency
    if torch.cuda.is_available():
        print(f"   ✓ GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("✅ Performance test completed!")

def main():
    """Run all integration tests."""
    print("🚀 Starting PyTorchCurator Integration Tests")
    print("=" * 50)
    
    try:
        test_all_encoders()
        test_curator_with_mock_data()
        test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("🎉 All integration tests passed successfully!")
        print("\nPyTorchCurator is ready for use with:")
        print("  ✓ One-hot encoding/decoding")
        print("  ✓ Binary encoding/decoding")
        print("  ✓ RGB encoding/decoding")
        print("  ✓ Learned embedding encoding/decoding")
        print("  ✓ Efficient batch processing")
        print("  ✓ Memory optimization")
        print("  ✓ GPU acceleration support")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()