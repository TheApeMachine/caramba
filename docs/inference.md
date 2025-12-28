# 🔮 Inference Guide

This guide covers text generation, KV-cache management, and speculative decoding in caramba. These features enable efficient inference with trained models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Standard Generation](#standard-generation)
- [KV-Cache Management](#kv-cache-management)
- [Auto Policy Selection](#auto-policy-selection)
- [Speculative Decoding](#speculative-decoding)
- [Decode Planning](#decode-planning)

---

## Overview

caramba's inference system provides:

- **Autoregressive generation** — Token-by-token text generation
- **KV-cache management** — Efficient storage and quantization
- **Automatic optimization** — Budget-aware cache selection
- **Speculative decoding** — Draft-verify acceleration

```python
from infer import Generator, GenerateConfig, generate

# Quick generation
output_ids = generate(model, input_ids, max_new_tokens=64)

# Configured generation
config = GenerateConfig(
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    cache_kind="auto",
)
generator = Generator(model, config=config)
output_ids = generator.generate(input_ids)
```

---

## Standard Generation

### Basic Usage

```python
from infer import generate, GenerateConfig

# Stateless generation (simplest)
output_ids = generate(
    model,
    input_ids,           # Shape: (B, T)
    max_new_tokens=64,
    temperature=0.8,
)

# Configured generation
config = GenerateConfig(
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    eos_token_id=2,
)
output_ids = generate(model, input_ids, config=config)
```

### Stateful Generator

For multi-turn generation or session reuse:

```python
from infer import Generator, GenerateConfig

config = GenerateConfig(
    max_new_tokens=128,
    max_seq_len=4096,
    cache_kind="fp16",
)

generator = Generator(model, config=config, device=device)

# First generation
output1 = generator.generate(input_ids_1)

# Continue with context
output2 = generator.generate(input_ids_2)

# Reset cache for new conversation
generator.reset()
```

### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 64 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature (0 = greedy) |
| `top_k` | int | None | Top-K sampling |
| `top_p` | float | None | Nucleus sampling |
| `eos_token_id` | int | None | Stop token |
| `max_seq_len` | int | 2048 | Maximum sequence length |

### Temperature Effects

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Greedy (deterministic) |
| 0.1-0.5 | Conservative sampling |
| 0.7-0.9 | Balanced creativity |
| 1.0+ | High diversity |

---

## KV-Cache Management

The KV-cache stores key and value tensors from previous tokens, enabling efficient autoregressive generation.

### Cache Kinds

caramba supports multiple cache quantization levels:

| Kind | Bytes/Token | Quality | Use Case |
|------|-------------|---------|----------|
| `fp32` | 4 | Perfect | Debugging |
| `fp16` | 2 | Near-perfect | Default inference |
| `q8` | 1 | Good | Memory-constrained |
| `q4` | 0.5 | Acceptable | Very long context |
| `nf4` | 0.5 | Good | Quantization research |

### Configuring Cache

```python
from infer import GenerateConfig
from config.kvcache import KVCacheKind

config = GenerateConfig(
    cache_kind=KVCacheKind.FP16,  # or "fp16"
    max_seq_len=4096,
)
```

### DBA Cache Compression

With Decoupled Bottleneck Attention, cache compression is even more dramatic:

| Configuration | Bytes/Token | Reduction |
|---------------|-------------|-----------|
| Standard (d=2048) | 2048 | 1× |
| DBA (sem=128, geo=256, fp16) | 384 | 5.3× |
| DBA (sem=128, geo=256, q8) | 192 | 10.7× |

### Manual Cache Configuration

```python
from cache import LayerKVCache, DecoupledLayerKVCache
from config.kvcache import KVCacheTensorConfig, KVCacheKind

# Standard cache
cache = LayerKVCache(
    batch_size=1,
    max_seq_len=2048,
    k_dim=256,
    v_dim=256,
    k_cfg=KVCacheTensorConfig(kind=KVCacheKind.Q8_0),
    v_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16),
    device=torch.device("mps"),
)

# Decoupled cache for DBA
cache = DecoupledLayerKVCache(
    batch_size=1,
    max_seq_len=2048,
    k_sem_dim=128,   # Semantic keys
    k_geo_dim=256,   # Geometric keys
    v_dim=256,
    k_sem_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16),
    k_geo_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16),
    v_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16),
    device=torch.device("mps"),
)
```

---

## Auto Policy Selection

When you don't know the best cache configuration, caramba can automatically select one based on constraints and quality gates.

### Enable Auto Selection

```python
config = GenerateConfig(
    cache_kind="auto",  # Let caramba choose
    cache_budget_mb=1024,  # Memory budget
)
```

### Selection Pipeline

```text
┌──────────────────────────────────────────────────────────┐
│               AUTO CACHE SELECTION                        │
├──────────────────────────────────────────────────────────┤
│  1. Budget Filter    Drop candidates > cache_budget_mb   │
│  2. Quality Gates    Test NLL, PPL, KL thresholds        │
│  3. Speed Pick       Benchmark remaining candidates      │
│  4. Persist          Save decision for reuse             │
└──────────────────────────────────────────────────────────┘
```

### Full Auto Configuration

```python
config = GenerateConfig(
    cache_kind="auto",

    # Budget constraint
    cache_budget_mb=1024,

    # Quality gates
    cache_quality_max_delta_nll=0.05,   # Max NLL increase
    cache_quality_max_ppl_ratio=1.05,   # Max PPL ratio
    cache_quality_max_mean_kl=0.1,      # Needle-in-haystack gate
    cache_quality_prompt_len=64,
    cache_quality_decode_steps=4,

    # Benchmarking
    cache_auto_benchmark=True,
    cache_auto_bench_steps=8,
    cache_auto_bench_prompt_len=64,

    # Persistence
    cache_plan_path="runs/cache_plans.json",
    cache_plan_probe=True,
    cache_plan_probe_interval_sec=3600,
)
```

### Quality Gates

| Gate | What it Checks |
|------|----------------|
| `cache_quality_max_delta_nll` | NLL difference from baseline |
| `cache_quality_max_ppl_ratio` | Perplexity ratio vs baseline |
| `cache_quality_max_mean_kl` | KL divergence (needle-in-haystack) |

### Plan Persistence

Auto-selected cache policies are persisted for reuse:

```python
config = GenerateConfig(
    cache_kind="auto",
    cache_plan_path="runs/cache_plans.json",  # Save decisions
    cache_plan_probe=True,  # Re-evaluate periodically
    cache_plan_probe_interval_sec=3600,  # Every hour
)
```

---

## Speculative Decoding

Accelerate inference by using a smaller draft model to propose tokens, then verify with the target model.

### Basic Speculative Decoding

```python
from infer import SpeculativeGenerator, SpeculativeConfig

config = SpeculativeConfig(
    spec_k=4,              # Draft 4 tokens per step
    max_new_tokens=128,
    temperature=0.8,
)

generator = SpeculativeGenerator(
    target_model=large_model,
    draft_model=small_model,
    config=config,
)

output = generator.generate(input_ids)
print(f"Acceptance rate: {generator.acceptance_rate:.2%}")
```

### Adaptive Speculative Decoding

Automatically adjust draft length based on acceptance rate:

```python
config = SpeculativeConfig(
    spec_k=4,
    spec_k_adaptive=True,          # Enable adaptive K
    spec_disable_below_accept=0.3, # Fall back if acceptance < 30%
    max_new_tokens=128,
)
```

### Speculative Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spec_k` | int | 4 | Tokens to draft per step |
| `spec_k_adaptive` | bool | False | Adjust K based on acceptance |
| `spec_method` | str | "reject_sampling" | Verification method |
| `spec_extra_token` | bool | True | Sample extra token on accept |
| `spec_disable_below_accept` | float | 0.0 | Threshold to fall back |

### Speedup Characteristics

| Acceptance Rate | Speedup |
|-----------------|---------|
| 90%+ | 2-3× |
| 70-90% | 1.5-2× |
| 50-70% | 1.2-1.5× |
| <50% | May slow down |

Best results when:
- Draft model is much smaller (e.g., 125M vs 7B)
- Task has predictable patterns (code completion, templates)
- Draft model shares vocabulary with target

---

## Decode Planning

For long-context inference, caramba can dynamically adjust memory usage patterns.

### Decode Plans

```python
config = GenerateConfig(
    decode_plan="auto",  # auto, fixed, none
)
```

| Plan | Behavior |
|------|----------|
| `auto` | Use buckets to pick q_chunk and local_window |
| `fixed` | Always use decode_q_chunk and decode_local_window |
| `none` | Use layer defaults |

### Bucket Configuration

```python
config = GenerateConfig(
    decode_plan="auto",

    # Bucket thresholds
    decode_bucket_short=512,
    decode_bucket_mid=2048,

    # Fixed values (for decode_plan="fixed")
    decode_q_chunk=128,
    decode_local_window=512,
)
```

### Long-Context Optimization

For very long sequences (8K+):

```python
config = GenerateConfig(
    max_seq_len=8192,
    cache_kind="q8",        # Quantized cache
    decode_plan="auto",     # Adaptive chunking
    cache_budget_mb=2048,   # Memory limit
)
```

---

## Programmatic Examples

### Complete Generation Pipeline

```python
import torch
from infer import Generator, GenerateConfig

# Load model
model = load_model("path/to/checkpoint")
model.eval()

# Configure generation
config = GenerateConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    max_seq_len=4096,
    cache_kind="fp16",
)

# Create generator
generator = Generator(model, config=config, device="mps")

# Generate
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = generator.generate(input_ids)
text = tokenizer.decode(output_ids[0])
print(text)
```

### Batch Generation

```python
# Multiple prompts
prompts = [
    "The weather today is",
    "In the year 2050,",
    "The secret to happiness is",
]

input_ids = tokenizer(prompts, return_tensors="pt", padding=True)
output_ids = generator.generate(input_ids["input_ids"])

for i, text in enumerate(tokenizer.batch_decode(output_ids)):
    print(f"Prompt {i}: {text}")
```

### Streaming Generation

```python
from infer import Generator

generator = Generator(model, config=config, device="mps")

# Token-by-token generation
input_ids = tokenizer.encode("Hello", return_tensors="pt")
for token_id in generator.generate_iter(input_ids):
    print(tokenizer.decode([token_id]), end="", flush=True)
```

---

## Manifest-Based Generation

Generation settings can be specified in manifests for reproducibility:

```yaml
targets:
  - type: experiment
    name: generate_test
    runs:
      - id: generate
        mode: generate
        generate:
          max_new_tokens: 128
          temperature: 0.8
          cache_kind: auto
          cache_budget_mb: 1024
```

---

## Summary

| Feature | Use Case | Key Config |
|---------|----------|------------|
| Standard | Basic generation | `GenerateConfig` |
| Cache Quantization | Memory savings | `cache_kind` |
| Auto Selection | Unknown constraints | `cache_kind="auto"` |
| Speculative | Speed boost | `SpeculativeConfig` |
| Decode Planning | Long context | `decode_plan="auto"` |

---

<div align="center">

**[← Training](training.md)** · **[Benchmarking →](benchmarking.md)**

</div>
