#!/usr/bin/env python3
"""Generate and visualize a single colored architectural sample."""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

from deepsculpt.core.data.generation.pytorch_sculptor import PyTorchSculptor
from deepsculpt.core.visualization.pytorch_visualization import PyTorchVisualizer


def main():
    void_dim = 32
    output_path = repo_root / "sample_preview.png"

    print(f"Generating architectural sample ({void_dim}^3)...")
    sculptor = PyTorchSculptor(
        void_dim=void_dim,
        edges=(3, 0.1, 0.9),
        planes=(3, 0.1, 0.9),
        pipes=(2, 0.1, 0.9),
        grid=(1, 4),
        device="cpu",
    )
    structure, colors = sculptor.generate_architectural_sculpture()

    # Convert to numpy for visualization
    if isinstance(structure, torch.Tensor):
        structure = structure.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    occupancy = (structure > 0).sum() / structure.size
    print(f"Structure shape: {structure.shape}, occupancy: {occupancy:.1%}")
    print(f"Colors shape: {colors.shape}, unique values: {np.unique(colors)}")

    # Map unique color values to distinct RGBA colors
    # Hash values change per Python session, so just assign by unique value
    # First color is gray (grid/columns/floors), rest are for primitives
    gray_hash = hash("gray") % 256
    palette = [
        (0.85, 0.25, 0.25, 1),   # red
        (0.20, 0.50, 0.85, 1),   # blue
        (0.95, 0.75, 0.20, 1),   # yellow
        (0.30, 0.75, 0.40, 1),   # green
        (0.80, 0.25, 0.75, 1),   # magenta
        (0.91, 0.50, 0.16, 1),   # orange
    ]

    rgba = np.zeros((*structure.shape, 4))

    # Map gray (grid/columns/floors) first
    rgba[colors == gray_hash] = (0.55, 0.55, 0.55, 1)

    # Map remaining colors to palette
    unique_vals = sorted([v for v in np.unique(colors) if v != 0 and v != gray_hash])
    print(f"Unique non-zero color indices (excluding gray): {unique_vals}")
    for i, val in enumerate(unique_vals):
        rgba[colors == val] = palette[i % len(palette)]

    # Only show filled voxels
    filled = structure > 0

    # matplotlib voxels() expects facecolors with shape matching the bool array
    # but only the True positions matter, so we can pass the full RGBA array
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig, axes = plt.subplots(2, 2, figsize=(16, 16),
                             subplot_kw={"projection": "3d"})
    fig.suptitle("Architectural Sample (colored)", fontsize=16, y=0.95)

    for i, ax in enumerate(axes.flat):
        rotated_struct = np.rot90(filled, i)
        rotated_colors = np.rot90(rgba, i)
        ax.voxels(rotated_struct, facecolors=rotated_colors,
                  edgecolors=rotated_colors * [0.7, 0.7, 0.7, 1], linewidth=0.3)
        ax.set_title(f"Rotation {i*90}°")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
