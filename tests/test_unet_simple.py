"""
Simple tests for 3D U-Net diffusion architecture with smaller memory footprint.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deepSculpt.pytorch_models import (
    SinusoidalPositionEmbedding,
    TimeEmbedding,
    ResidualBlock3D,
    UNet3D,
    ConditionalUNet3D
)

def test_sinusoidal_embedding():
    """Test sinusoidal position embedding."""
    print("Testing SinusoidalPositionEmbedding...")
    
    embedding = SinusoidalPositionEmbedding(128)
    timesteps = torch.tensor([0, 100, 500])
    
    output = embedding(timesteps)
    assert output.shape == (3, 128)
    print("✓ SinusoidalPositionEmbedding shape test passed")

def test_time_embedding():
    """Test time embedding module."""
    print("Testing TimeEmbedding...")
    
    time_emb = TimeEmbedding(128, 256)
    timesteps = torch.tensor([100, 200])
    
    output = time_emb(timesteps)
    assert output.shape == (2, 256)
    print("✓ TimeEmbedding shape test passed")

def test_residual_block():
    """Test 3D residual block."""
    print("Testing ResidualBlock3D...")
    
    block = ResidualBlock3D(32, 64, 128)
    
    x = torch.randn(1, 32, 8, 8, 8)
    time_emb = torch.randn(1, 128)
    
    output = block(x, time_emb)
    assert output.shape == (1, 64, 8, 8, 8)
    print("✓ ResidualBlock3D shape test passed")

def test_unet_small():
    """Test complete U-Net with small dimensions."""
    print("Testing UNet3D with small dimensions...")
    
    # Use very small dimensions to avoid memory issues
    unet = UNet3D(
        in_channels=2, 
        out_channels=2, 
        base_channels=16,  # Very small base channels
        num_levels=2,      # Only 2 levels
        use_attention=False  # Disable attention to save memory
    )
    
    # Very small input
    x = torch.randn(1, 2, 8, 8, 8)
    timesteps = torch.tensor([100])
    
    output = unet(x, timesteps)
    assert output.shape == (1, 2, 8, 8, 8)
    print("✓ UNet3D small test passed")

def test_conditional_unet_small():
    """Test conditional U-Net with small dimensions."""
    print("Testing ConditionalUNet3D with small dimensions...")
    
    unet = ConditionalUNet3D(
        in_channels=2,
        out_channels=2,
        base_channels=16,
        num_levels=2,
        use_attention=False,
        use_cross_attention=False  # Disable cross-attention to save memory
    )
    
    # Check if the forward method has the expected signature
    print(f"ConditionalUNet3D forward method: {unet.forward}")
    
    x = torch.randn(1, 2, 8, 8, 8)
    timesteps = torch.tensor([100])
    
    # Test without conditioning first
    output = unet(x, timesteps)
    assert output.shape == (1, 2, 8, 8, 8)
    print("✓ ConditionalUNet3D without conditioning test passed")
    
    # Test with conditioning
    try:
        text_emb = torch.randn(1, 512)
        output = unet(x, timesteps, text_emb=text_emb)
        assert output.shape == (1, 2, 8, 8, 8)
        print("✓ ConditionalUNet3D with conditioning test passed")
    except Exception as e:
        print(f"Conditioning test failed: {e}")
        # Try with positional arguments
        output = unet.forward(x, timesteps, text_emb)
        assert output.shape == (1, 2, 8, 8, 8)
        print("✓ ConditionalUNet3D with positional args test passed")

def test_gradient_flow():
    """Test that gradients flow through the network."""
    print("Testing gradient flow...")
    
    unet = UNet3D(2, 2, 16, num_levels=2, use_attention=False)
    
    x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    timesteps = torch.tensor([100])
    
    output = unet(x, timesteps)
    loss = output.sum()
    loss.backward()
    
    # Check that input has gradients
    assert x.grad is not None
    print("✓ Gradient flow test passed")

def test_memory_usage():
    """Test memory usage reporting."""
    print("Testing memory usage reporting...")
    
    unet = UNet3D(2, 2, 16, num_levels=2)
    
    memory_stats = unet.get_memory_usage()
    
    assert "total_parameters" in memory_stats
    assert "trainable_parameters" in memory_stats
    assert memory_stats["total_parameters"] > 0
    print("✓ Memory usage reporting test passed")

if __name__ == "__main__":
    print("Running simple 3D U-Net tests...")
    
    try:
        test_sinusoidal_embedding()
        test_time_embedding()
        test_residual_block()
        test_unet_small()
        test_conditional_unet_small()
        test_gradient_flow()
        test_memory_usage()
        
        print("\n🎉 All simple U-Net tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()