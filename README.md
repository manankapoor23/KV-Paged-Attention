# KV-Paged Inference System for Transformer Models

A systems-focused exploration of how modern LLM inference engines manage memory efficiently during autoregressive decoding.

## 1. Motivation
Well , the motivation for this project basically came when I read the paper "Attention Is All You Need" and OH MY GOD , it gave me all the motivation to think about LLMs and then randomly one day I was scrolling on X and read about vLLMs and what they do , and now here we are.
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

## 9. Repository Structure

```
KV-Paged/
├── pages/                          # Core systems implementation
│   ├── page.py                     # KVPage abstraction — fixed-size KV storage
│   ├── page_pool.py                # PagePool allocator — free/used page management
│   ├── page_table.py               # PageTable — logical token → physical mapping
│   ├── paged_kv_reader.py          # KV gathering from non-contiguous pages
│   ├── prefix_cache.py             # PrefixCache — prefix reuse mechanism
│   ├── attention.py                # Paged attention execution
│   ├── driver_day5.py              # End-to-end inference simulation
│   ├── test.py                     # Core unit tests
│   ├── test_day3.py                # Test suite — day 3 iterations
│   └── test_day4.py                # Test suite — day 4 iterations
│
├── comparison/                     # Naive vs paged attention validation
│   ├── naive_attention.py          # Baseline attention (contiguous KV)
│   ├── paged_attention.py          # Paged attention (indirect KV access)
│   ├── driver_day4.py              # Comparison & correctness validation
│   └── blah_blah.txt               # Notes
│
├── Benchmarks/                     # Reference implementations
│   └── naive-kv-cache.py           # Naive KV cache baseline
│
├── reuseable/                      # Utility & reusable components
│   └── reuse_core.txt
│
├── kv_tensor_visualization.py      # Visualization & debugging tools
├── test.ipynb                      # Notebook for experiments
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── venv/                           # Virtual environment
```

## 10. Key Files & Their Role

| File | Purpose |
|------|---------|
| [pages/page.py](pages/page.py) | KVPage data structure with K, V tensors and reference counting |
| [pages/page_pool.py](pages/page_pool.py) | Memory allocator managing page lifecycle |
| [pages/page_table.py](pages/page_table.py) | Virtual memory mapping (token index → page + slot) |
| [pages/attention.py](pages/attention.py) | Scaled dot-product attention kernel |
| [pages/driver_day5.py](pages/driver_day5.py) | Multi-request inference simulation |
| [comparison/naive_attention.py](comparison/naive_attention.py) | Baseline for correctness validation |
| [comparison/paged_attention.py](comparison/paged_attention.py) | Paged variant for direct comparison |
