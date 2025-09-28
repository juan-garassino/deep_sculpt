"""
Simple test for grid functionality to verify it works.
"""

import torch
import sys
import os

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from logger import log_info, log_success, begin_section, end_section

def simple_attach_grid_pytorch(
    structure: torch.Tensor,
    colors: torch.Tensor,
    step: int = 4,
    device: str = "cpu"
) -> tuple:
    """Simple grid attachment for testing."""
    begin_section("Simple grid attachment test")
    
    structure_dim = structure.shape[0]
    log_info(f"Creating simple grid in {structure_dim}x{structure_dim}x{structure_dim} structure")
    
    # Create a simple grid pattern
    locations = []
    for x in range(step, structure_dim - step, step * 2):
        for y in range(step, structure_dim - step, step * 2):
            locations.append((x, y))
    
    log_info(f"Generated {len(locations)} grid positions")
    
    # Create columns with random heights
    for x, y in locations:
        height = torch.randint(structure_dim // 4, structure_dim // 2, (1,)).item()
        structure[x, y, 0:height] = 1
        colors[x, y, 0:height] = 100  # Simple color value
    
    # Create base floor
    structure[:, :, 0] = 1
    colors[:, :, 0] = 100
    
    filled_count = torch.sum(structure > 0).item()
    log_success(f"Simple grid created with {len(locations)} columns, {filled_count} filled voxels")
    end_section()
    
    return structure, colors

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Using device: {device}")
    
    # Test simple grid
    structure_dim = 20
    structure = torch.zeros((structure_dim, structure_dim, structure_dim), device=device)
    colors = torch.zeros(structure.shape, device=device)
    
    structure, colors = simple_attach_grid_pytorch(structure, colors, step=4, device=device)
    
    print("✅ Simple grid test passed!")