#!/usr/bin/env python3

import torch

# Test tensor permutation
print("Testing tensor permutation...")

# Create test tensor in TensorFlow format: [batch, depth, height, width, channels]
x = torch.randn(2, 32, 32, 32, 3)
print(f"Original shape (TF format): {x.shape}")

# Apply the permute operation from SimpleDiscriminator
x_permuted = x.permute(0, 4, 1, 2, 3)
print(f"After permute(0, 4, 1, 2, 3): {x_permuted.shape}")

# This should give us [batch, channels, depth, height, width] = [2, 3, 32, 32, 32]
print(f"Expected: [2, 3, 32, 32, 32]")
print(f"Got: {list(x_permuted.shape)}")
print(f"Correct: {list(x_permuted.shape) == [2, 3, 32, 32, 32]}")