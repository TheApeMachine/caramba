# ⚡ Optimization Guide

caramba includes multiple optimization layers to maximize performance on your hardware. This guide covers fused kernels, runtime planning, and platform-specific optimizations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Runtime Planning](#runtime-planning)
- [Metal Kernels (Apple Silicon)](#metal-kernels-apple-silicon)
- [Triton Kernels (CUDA)](#triton-kernels-cuda)
- [torch.compile](#torchcompile)
- [Memory Optimization](#memory-optimization)
- [Platform Comparison](#platform-comparison)

---

## Overview

caramba optimizes at multiple levels:

| Level | What | How |
|-------|------|-----|
| **Runtime Planning** | Batch size, dtype, AMP | Cached decisions |
| **Kernel Fusion** | Attention, normalization | Metal/Triton |
| **Compilation** | Graph optimization | torch.compile |
| **Memory** | Activation checkpointing, offload | Config flags |

The optimization philosophy:
- **Config is declarative** — You specify intent
- **Runtime is adaptive** — caramba makes measured decisions
- **Results are cached** — Repeated runs reuse optimizations

---

## Runtime Planning

caramba caches optimization decisions based on a signature of your configuration:

### What Gets Cached

```text
Signature = (device + manifest + train_config)
           ↓
RuntimePlan:
  - dtype / AMP dtype
  - batch_size (with auto-scaling)
  - torch.compile enabled + mode
  - other runtime knobs
```

### Auto-Fit Features

#### Batch Size Auto-Scaling

When training with `batch_size: auto`:

1. Start with configured batch size
2. Profile memory usage
3. Scale up if memory available
4. Scale based on `block_size` when appropriate

#### Dtype Auto-Selection

When `dtype: auto`:

```yaml
train:
  dtype: auto  # Let caramba choose
```

Selection logic:
- Check device capabilities (fp16, bf16 support)
- Consider training phase (blockwise vs global)
- Balance precision vs speed

#### AMP Auto-Configuration

```yaml
train:
  use_amp: auto
  amp_dtype: auto  # float16 or bfloat16
```

### Plan Persistence

Runtime plans are cached in `caramba/runtime/plan/` (see `caramba/runtime/plan/__init__.py`):

```python
runtime.plan import RuntimePlan, load_plan, save_plan

# Plans keyed by signature
plan = load_plan(signature)
if plan is None:
    plan = compute_optimal_plan(...)
    save_plan(signature, plan)
```

### Force Recomputation

```bash
# Delete cached plans to force recomputation
rm -rf .caramba/runtime_plans/
```

---

## Metal Kernels (Apple Silicon)

caramba includes optimized Metal kernels for Apple Silicon (M1/M2/M3/M4).

### What's Optimized

| Kernel | Operation | Speedup |
|--------|-----------|---------|
| `dba_decode.metal` | Fused DBA attention decode | 2-5× |
| `rmsnorm.metal` | RMS normalization | 1.5-2× |
| `layernorm.metal` | Layer normalization | 1.5× |
| `rope.metal` | Rotary embeddings | 1.5× |
| `lion.metal` | Lion optimizer | 1.5× |

### DBA Decode Fusion

The flagship optimization for DBA inference:

```text
Standard Path:
  Q_sem·K_sem^T → store → Q_geo·K_geo^T → store → add → softmax → V

Fused Path:
  (Q_sem·K_sem^T + Q_geo·K_geo^T) → softmax → V  [single kernel]
```

Benefits:
- Eliminates intermediate storage
- Reduces kernel launch overhead
- Uses online softmax for numerical stability

### When Metal Kernels Activate

Automatic activation when:
- Device is `mps`
- Model uses `AttentionMode.DECOUPLED`
- Decode step (`T == 1`)
- KV-cache is `fp16`
- Xcode Command Line Tools available

### Force-Building Metal Extension

```python
optimizer.metal.jit import load_caramba_metal_ops

# Build and load Metal extension
ops = load_caramba_metal_ops(verbose=True)
```

### Requirements

- macOS with Apple Silicon
- Xcode Command Line Tools (`xcrun` available)
- PyTorch with MPS support

Check availability:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Fallback Behavior

If Metal kernels fail to compile:
- caramba logs a warning
- Falls back to PyTorch operations
- No functionality loss, only performance

---

## Triton Kernels (CUDA)

For NVIDIA GPUs, caramba uses Triton kernels for fused operations.

### What's Optimized

| Kernel | Operation | Features |
|--------|-----------|----------|
| Fused Attention | Decoupled attention decode | Quantized cache support |
| Split-K | Long-context attention | 2-pass for memory efficiency |
| Quantized Ops | Cache dequantization | Q4/Q8/NF4 |

### Decoupled Attention Decode

```python
optimizer.triton_runtime import TRITON_AVAILABLE
optimizer.fused_attention import fused_decode_available

if TRITON_AVAILABLE and fused_decode_available(cache, "cuda"):
    # Will use fused kernel automatically
    pass
```

Features:
- Fuses dequantization + attention + softmax
- FlashAttention-style online softmax
- Supports Q4/Q8/NF4 quantized caches
- Split-K for very long prefixes

### Requirements

- NVIDIA GPU with CUDA
- Triton installed (`pip install triton`)
- CUDA toolkit

Check availability:

```python
optimizer.triton_runtime import TRITON_AVAILABLE
print(f"Triton available: {TRITON_AVAILABLE}")
```

---

## torch.compile

caramba supports PyTorch 2.0's `torch.compile` for graph optimization.

### Enable Compilation

```yaml
train:
  compile_model: true
```

Or auto-detect:

```yaml
train:
  compile_model: auto  # Enable if beneficial
```

### Compile Modes

| Mode | Tradeoff |
|------|----------|
| `default` | Balanced compile time vs speedup |
| `reduce-overhead` | Minimize kernel launch overhead |
| `max-autotune` | Maximum optimization (slow compile) |

### When to Use

✅ **Enable for:**
- Production inference
- Long training runs
- Compute-bound workloads

❌ **Avoid for:**
- Quick experiments
- Debugging
- Frequently changing models

### Programmatic Usage

```python
import torch

model = build_model(...)
model = torch.compile(model, mode="reduce-overhead")
```

---

## Memory Optimization

### Activation Checkpointing

Trade compute for memory by recomputing activations during backward:

```yaml
train:
  activation_checkpointing: true
  activation_checkpoint_threshold: 0.9  # Trigger at 90% memory
```

### Gradient Accumulation

Effective larger batch sizes without more memory:

```yaml
train:
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 32
```

### Mixed Precision Training

Reduce memory with half-precision:

```yaml
train:
  use_amp: true
  amp_dtype: float16  # or bfloat16
```

### KV-Cache Quantization

Reduce inference memory:

```yaml
# In benchmarks or generation
cache_kind: q8  # or q4 for more savings
```

### Teacher Output Caching

For upcycle training, cache teacher outputs:

```yaml
train:
  phase: blockwise
  cache_teacher_outputs: true
```

Benefits:
- Avoid repeated teacher forward passes
- Significant speedup for blockwise distillation
- Trades memory for speed

---

## Platform Comparison

### Apple Silicon vs NVIDIA A100

| Aspect | Apple Silicon (M4 Max) | A100 80GB |
|--------|------------------------|-----------|
| **Memory** | 128GB unified | 80GB HBM |
| **Bandwidth** | ~400 GB/s | ~2 TB/s |
| **Compute** | Lower | Higher |
| **Workload fit** | Larger models (fits) | Faster throughput |
| **Best for** | Iteration, fitting | Production training |

### When Apple Silicon Wins

- **Unified memory** — Fit workloads that OOM on 80GB
- **Local iteration** — Fast experiment turnaround
- **Inference** — Reasonable throughput for demos

### What caramba Optimizes for Apple Silicon

1. **Fewer memory round-trips** — Kernel fusion
2. **Fewer launches** — Reduce framework overhead
3. **UMA-friendly workflows** — mmap datasets, efficient state handling

### Configuration by Platform

#### Apple Silicon (MPS)

```yaml
train:
  device: mps
  dtype: float32        # fp16 can be unstable
  use_amp: false        # MPS AMP is limited
  compile_model: false  # Limited compile support
  num_workers: 0        # MPS prefers main process
```

#### NVIDIA CUDA

```yaml
train:
  device: cuda
  dtype: bfloat16       # Best for Ampere+
  use_amp: true
  amp_dtype: bfloat16
  compile_model: true
  num_workers: 4
  pin_memory: true
```

---

## Optimizer Orchestration

Beyond kernel optimization, caramba optimizes the training process itself.

### Dynamic Strategy Switching

The orchestrator monitors training and switches optimizers:

```yaml
train:
  orchestrator_enabled: true
  orchestrator_decision_interval: 500
  orchestrator_initial_strategy: conservative_adamw
```

See [Training Guide → Orchestrated Training](training.md#orchestrated-training) for details.

### Built-in Components

| Component | Purpose |
|-----------|---------|
| **AdaGC** | Per-parameter adaptive gradient clipping |
| **SWATS** | Auto-switch Adam → SGD when stable |
| **PIDAO** | PID-controller optimizer |
| **Nowcasting** | Predict weights to skip steps |

---

## Profiling and Debugging

### Check What's Being Used

```python
optimizer.runtime import (
    TRITON_AVAILABLE,
    METAL_AVAILABLE,
    get_backend_info,
)

print(get_backend_info())
# {
#   'triton': False,
#   'metal': True,
#   'metal_dba': True,
#   'compile': True,
# }
```

### Profile Memory

```python
import torch

# Track memory usage
torch.mps.empty_cache()  # or torch.cuda.empty_cache()
print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
```

### Profile Kernels

```python
# Time individual operations
import time

start = time.perf_counter()
output = model(input_ids)
torch.mps.synchronize()  # or torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"Forward: {elapsed*1000:.2f} ms")
```

---

## Summary

| Optimization | Platform | Activation |
|--------------|----------|------------|
| Runtime Planning | All | Automatic |
| Metal Kernels | MPS | Automatic (when available) |
| Triton Kernels | CUDA | Automatic (when available) |
| torch.compile | CUDA (best) | `compile_model: true` |
| Activation Checkpointing | All | `activation_checkpointing: true` |
| Mixed Precision | CUDA (best) | `use_amp: true` |
| KV-Cache Quantization | All | `cache_kind: q8/q4` |

caramba's optimization approach:
- **Declarative config** — You specify what you want
- **Adaptive runtime** — caramba decides how
- **Cached decisions** — Fast repeated runs
- **Graceful fallback** — Always works, just faster with optimizations

---

<div align="center">

**[← Agents](agents.md)** · **[Back to README](../README.md)**

</div>
