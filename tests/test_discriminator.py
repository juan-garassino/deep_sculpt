#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.models.model_factory import PyTorchModelFactory

def test_discriminator():
    print("Testing discriminator directly...")
    
    # Create discriminator
    factory = PyTorchModelFactory()
    disc = factory.create_gan_discriminator('simple', void_dim=32, color_mode=0)
    
    print(f"Discriminator: {disc}")
    print(f"Input channels: {disc.input_channels}")
    print(f"Conv1: {disc.conv1}")
    
    # Create test tensor in PyTorch format
    x = torch.randn(2, 1, 32, 32, 32)  # [batch, channels, depth, height, width]
    print(f"Input tensor shape: {x.shape}")
    
    try:
        # Call discriminator
        output = disc(x)
        print(f"Output shape: {output.shape}")
        print("SUCCESS: Discriminator worked correctly!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_discriminator()