# KV-Paged Inference System for Transformer Models

A systems-focused exploration of how modern LLM inference engines manage memory efficiently during autoregressive decoding.

## 1. Motivation

Large Language Models rely on a Key-Value (KV) cache to accelerate autoregressive inference.
A naïve KV cache grows linearly with sequence length, requires contiguous memory, and becomes inefficient or infeasible for:

- long contexts
- multi-turn chat
- concurrent inference requests

Modern inference engines (e.g., vLLM, TGI) solve this using paged KV caching, but most explanations stop at a high level.

**Goal of this project:**

Understand and implement the core memory and execution mechanisms behind paged KV caching and paged attention — from first principles.

This is a systems design project, not a training or benchmarking exercise , yes but surely during the coding part of this , I had to learn the math , and everything behind how it works and then proceed to it.

## 2. Problem Statement
### Initial approach: Naïve KV Cache

**Design**
- Store KV tensors contiguously per request (Contigous Memory , ts is too bad)
- Append KV for each new token
- Recompute attention using full prefix

**Problems encountered**
- KV memory grows unbounded with sequence length
- Requires contiguous memory allocation
- Memory fragmentation(yup , my OS course finally helped me) under concurrent requests
- No safe way to share prefixes across requests
- Copying KV for branching requests is expensive

This mirrors the limitations of early transformer inference implementations.

## 3. Design Constraints

This project intentionally operates under realistic inference constraints:

- CPU-only execution (no CUDA / Triton)
- Single-process, multi-request simulation
- Correctness > throughput
- Explicit memory ownership and lifecycle
- No modification of HuggingFace internals

These constraints force systems-level design decisions, not library shortcuts.

## 4. Core Design: Paged KV Cache
### Key idea

Decouple logical token order from physical KV storage.

Instead of storing KV contiguously, KV is stored in fixed-size pages, similar to OS virtual memory.

Few questions came to my mind ->
What if I just let the KV cache grow forever? -> it will eventually run out of memory
Solving the memory issue with normal KV caching is going to be a pain in the ass.

### Components

**KVPage**
- Fixed-size container holding KV slots across layers and heads.

**PagePool (Allocator)**
- Manages free/used pages with explicit lifecycle control.

**PageTable**
- Maps logical token index → (page_id, slot).

**PrefixCache**
- Enables reuse of KV pages for shared prefixes across requests.

**Reference Counting + Copy-on-Write**
- Allows safe sharing of KV pages while preventing data corruption during divergence.

This design removes the need for contiguous KV memory and enables efficient prefix reuse.

## 5. Attention Execution
### Baseline: Naïve Attention

- Reads KV from contiguous tensors
- O(n²) attention computation per token
- Simple but tightly coupled to memory layout

### Implemented: Paged Attention

- Reads KV indirectly via page tables
- KV gathered across non-contiguous pages
- Attention math unchanged
- Execution decoupled from storage layout

**Correctness guarantee:**
Paged attention output is numerically equivalent to naive attention (validated empirically).

## 6. Trade-Offs and Design Decisions
### Why paging?

- Avoids contiguous allocation
- Enables prefix reuse
- Matches production inference architectures

### Why not integrate into HF attention kernels?

- Requires custom CUDA/Triton kernels
- Out of scope for correctness-first systems exploration

### Why CPU-only?

- Forces explicit reasoning about memory and execution (my main was focus was to know how it works)
- Avoids hiding complexity behind GPU kernels

### Why visualization?

- Inference systems are hard to reason about without visibility
- Debugging and teaching require introspection

## 7. What This Project Is Not

 A training pipeline

 A performance benchmark

 A replacement for vLLM (I AM NOT EVEN CLOSE)

 A chat application 

This is a reference implementation focused on understanding inference-time systems design.

## 8. Measured Properties

While not optimized for throughput, the system explicitly tracks:

- KV memory growth behavior
- Page allocation / reuse events
- Copy-on-write triggers
- Prefix reuse effectiveness
- Numerical equivalence with naive attention

These are the metrics that matter for inference correctness and scalability, not accuracy scores.

## The Repo Structure 
phew this repo I build , finally we are here , I mean the structure.
KV-Paged/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
│
├── pages/                             # Core systems implementation
│   ├── page.py                        # KVPage abstraction
│   ├── page_pool.py                   # Memory allocator & lifecycle
│   ├── page_table.py                  # Logical → physical mapping
│   ├── paged_kv_reader.py             # KV gathering from pages
│   ├── prefix_cache.py                # Prefix reuse mechanism
│   ├── attention.py                   # Paged attention execution
│   ├── driver_day5.py                 # End-to-end inference driver
│   ├── test.py                        # Unit tests
│   ├── test_day3.py                   # Test suite day 3
│   ├── test_day4.py                   # Test suite day 4
│   └── __init__.py
│
├── comparison/                        # Naive vs paged attention
│   ├── naive_attention.py             # Baseline attention
│   ├── paged_attention.py             # Paged attention implementation
│   ├── driver_day4.py                 # Comparison driver
│   └── blah_blah.txt
│
├── Benchmarks/                        # Baseline implementations
│   └── naive-kv-cache.py              # Naive KV cache reference
│
├── reuseable/                         # Reusable utilities
│   └── reuse_core.txt
│
├── root-level utilities/
│   ├── kv_tensor_visualization.py     # Visualization tools
│   ├── test.ipynb                     # Notebook experiments
│   ├── core_issue.txt                 # Problem notes
│   ├── day_1.txt                      # Development notes
│   └── venv/                          # Python virtual environment