#!/usr/bin/env python3
"""Quick test to see what's failing in PyTorchSculptor"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

try:
    from core.data.generation.pytorch_sculptor import PyTorchSculptor
    
    print("Creating sculptor...")
    sculptor = PyTorchSculptor(
        void_dim=16,
        edges=(1, 0.3, 0.5),
        planes=(1, 0.3, 0.5),
        pipes=(1, 0.3, 0.5),
        grid=(1, 4),
        device='cpu',
        sparse_mode=False,
        verbose=True,  # Enable verbose to see errors
    )
    
    print("Generating sculpture...")
    structure, colors = sculptor.generate_sculpture()
    
    print(f"✅ Success! Generated structure: {structure.shape}")
    print(f"   Non-zero voxels: {(structure > 0).sum().item()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
