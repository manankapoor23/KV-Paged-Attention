"""
Visualize Raw KV Cache Tensors
Shows the structure and values of key and value tensors
"""

import torch
import numpy as np

# Configuration
batch_size = 2
seq_length = 5
hidden_dim = 64
num_heads = 8

print("=" * 70)
print("RAW KV CACHE TENSOR VISUALIZATION")
print("=" * 70)

# Create sample KV tensors
print("\n1. BASIC TENSOR SHAPES")
print("-" * 70)

# Key and Value tensors: shape (batch, seq_len, hidden_dim)
K = torch.randn(batch_size, seq_length, hidden_dim)
V = torch.randn(batch_size, seq_length, hidden_dim)

print(f"Key (K) tensor shape: {K.shape}")
print(f"  - Batch size: {batch_size}")
print(f"  - Sequence length: {seq_length} tokens")
print(f"  - Hidden dimension: {hidden_dim}")

print(f"\nValue (V) tensor shape: {V.shape}")
print(f"  - Same structure as Key tensor")

# Show actual values
print("\n\n2. SAMPLE VALUES")
print("-" * 70)

print(f"First batch, first 3 tokens, first 8 dimensions of Key:")
print(K[0, :3, :8])

print(f"\n\nFirst batch, first 3 tokens, first 8 dimensions of Value:")
print(V[0, :3, :8])

# Multi-head format
print("\n\n3. MULTI-HEAD ATTENTION FORMAT")
print("-" * 70)
head_dim = hidden_dim // num_heads
K_multihead = K.view(batch_size, seq_length, num_heads, head_dim)
V_multihead = V.view(batch_size, seq_length, num_heads, head_dim)

print(f"Key (multi-head) shape: {K_multihead.shape}")
print(f"  - Batch: {batch_size}")
print(f"  - Sequence length: {seq_length} tokens")
print(f"  - Number of heads: {num_heads}")
print(f"  - Dimensions per head: {head_dim}")

print(f"\nValue (multi-head) shape: {V_multihead.shape}")

print(f"\nFirst batch, first token, all 8 heads, first 4 dims per head:")
print(K_multihead[0, 0, :, :4])

# Batched operations
print("\n\n4. BATCHED KV TENSORS")
print("-" * 70)

print(f"Batch 0 shape: {K[0].shape} (one sequence of {seq_length} tokens)")
print(f"Batch 1 shape: {K[1].shape} (one sequence of {seq_length} tokens)")

print(f"\nSecond batch, first token values (first 16 dims):")
print(K[1, 0, :16])

# Memory information
print("\n\n5. MEMORY INFORMATION")
print("-" * 70)

k_memory = K.element_size() * K.nelement() / (1024 * 1024)
v_memory = V.element_size() * V.nelement() / (1024 * 1024)
total_memory = (k_memory + v_memory)

print(f"Key tensor size: {k_memory:.4f} MB")
print(f"Value tensor size: {v_memory:.4f} MB")
print(f"Total KV cache size: {total_memory:.4f} MB")

print(f"\nPer token memory:")
memory_per_token = (K.element_size() * hidden_dim * 2) / 1024  # K + V
print(f"  - {memory_per_token:.2f} KB per token (both K and V)")

print(f"\nDtype: {K.dtype}")
print(f"Device: {K.device}")

print("\n" + "=" * 70)
