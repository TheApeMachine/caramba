# 🧱 Layer Reference

Layers are the computational building blocks in caramba. Each layer is a thin PyTorch module that performs one specific operation. This document covers all available layer types and their configurations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Attention](#attention)
- [Feed-Forward](#feed-forward)
- [Mixture of Experts](#mixture-of-experts)
- [State Space Models](#state-space-models)
- [Normalization](#normalization)
- [Embeddings](#embeddings)
- [LoRA](#lora)
- [Utility Layers](#utility-layers)
- [Diffusion Head](#diffusion-head)

---

## Overview

Layers are specified in topology configurations:

```yaml
topology:
  type: StackedTopology
  layers:
    - type: AttentionLayer
      d_model: 512
      n_heads: 8
      mode: standard
```

All layers share common patterns:
- **Config-driven** — Parameters come from typed config objects
- **Composable** — Layers combine via topologies
- **Thin** — One concept per file, minimal abstraction

---

## Attention

The `AttentionLayer` supports three attention modes, covering the full spectrum from standard multi-head attention to compressed bottleneck variants.

### Standard Attention

Classic multi-head attention where every head has its own K/V projections:

```yaml
- type: AttentionLayer
  d_model: 512
  n_heads: 8
  mode: standard
  rope_enabled: true
  is_causal: true
  dropout_p: 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Model dimension |
| `n_heads` | int | required | Number of attention heads |
| `mode` | string | `standard` | Attention mode |
| `rope_enabled` | bool | `false` | Enable rotary position embeddings |
| `rope_base` | float | `10000.0` | RoPE base frequency |
| `is_causal` | bool | `true` | Use causal (autoregressive) masking |
| `dropout_p` | float | `0.0` | Attention dropout probability |

### Grouped-Query Attention (GQA)

Fewer KV heads shared across query heads, reducing KV-cache size:

```yaml
- type: AttentionLayer
  d_model: 2048
  n_heads: 32
  n_kv_heads: 8          # 4:1 sharing ratio
  mode: gqa
  rope_enabled: true
  is_causal: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_kv_heads` | int | `n_heads` | Number of KV heads (must divide `n_heads`) |

### Decoupled Bottleneck Attention (DBA)

Splits attention into semantic (content) and geometric (position) paths:

```yaml
- type: AttentionLayer
  d_model: 2048
  n_heads: 32
  n_kv_heads: 8
  mode: decoupled
  sem_dim: 128           # Semantic bottleneck (no RoPE)
  geo_dim: 256           # Geometric bottleneck (with RoPE)
  rope_enabled: true
  rope_base: 500000.0
  is_causal: true
  decoupled_gate: true   # Learnable gating between paths
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sem_dim` | int | 128 | Semantic path dimension (content) |
| `geo_dim` | int | 256 | Geometric path dimension (position) |
| `decoupled_gate` | bool | `false` | Enable learnable semantic/geometric gating |

**KV-Cache Comparison:**

| Mode | KV-Cache per Token | Reduction |
|------|-------------------|-----------|
| Standard (d=2048) | 2048 bytes | 1× |
| GQA (8 KV heads) | 512 bytes | 4× |
| DBA (sem=128, geo=256) | 384 bytes | 5.3× |

### Attention in Practice

Typical transformer block with attention:

```yaml
- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: AttentionLayer
      d_model: ${d_model}
      n_heads: ${n_heads}
      n_kv_heads: ${n_kv_heads}
      mode: decoupled
      sem_dim: 128
      geo_dim: 256
      rope_enabled: true
```

---

## Feed-Forward

### SwiGLU Layer

The default feed-forward network using SwiGLU activation:

```yaml
- type: SwiGLULayer
  d_model: 512
  d_ff: 2048
  bias: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Input/output dimension |
| `d_ff` | int | required | Hidden dimension (typically 4× d_model) |
| `bias` | bool | `false` | Include bias in linear layers |

**How SwiGLU works:**

```
output = down(silu(gate(x)) * up(x))
```

### GLU Variants

The `GLULayer` supports multiple gating mechanisms:

```yaml
- type: GLULayer
  d_model: 512
  d_ff: 2048
  activation: geglu      # swiglu, geglu, reglu
```

| Variant | Activation | Description |
|---------|------------|-------------|
| `swiglu` | SiLU (Swish) | Default, used in Llama |
| `geglu` | GELU | Used in some T5 variants |
| `reglu` | ReLU | Simpler but effective |

---

## Mixture of Experts

The `MoELayer` replaces dense feed-forward with sparse expert routing:

```yaml
- type: MoELayer
  d_model: 512
  num_experts: 8         # Total experts
  top_k: 2               # Active experts per token
  d_ff: 2048             # Per-expert hidden dim
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Model dimension |
| `num_experts` | int | required | Total number of experts |
| `top_k` | int | required | Experts activated per token |
| `d_ff` | int | required | Hidden dimension per expert |
| `bias` | bool | `false` | Include bias in projections |

### MoE Transformer Block

```yaml
- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: AttentionLayer
      d_model: ${d_model}
      n_heads: ${n_heads}
      mode: standard

- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: MoELayer
      d_model: ${d_model}
      num_experts: 8
      top_k: 2
      d_ff: ${d_ff}
```

### Load Balancing

MoE includes auxiliary loss for load balancing:

```python
# Access during training
aux_loss = moe_layer.aux_loss
total_loss = main_loss + 0.01 * aux_loss
```

The layer also tracks expert utilization:

```python
# Expert load (EMA of token distribution)
print(moe_layer.expert_load)  # tensor([0.12, 0.13, 0.11, ...])
```

---

## State Space Models

The `SSMLayer` implements Mamba-style selective state space models:

```yaml
- type: SSMLayer
  d_model: 512
  d_state: 16            # State dimension
  d_conv: 4              # Convolution kernel size
  expand: 2              # Expansion factor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Input/output dimension |
| `d_state` | int | `16` | Recurrent state dimension |
| `d_conv` | int | `4` | 1D convolution kernel size |
| `expand` | int | `2` | Inner dimension multiplier |
| `dt_rank` | str/int | `"auto"` | Rank of Δ projection |
| `bias` | bool | `false` | Include bias in projections |

### SSM Block

```yaml
- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: SSMLayer
      d_model: ${d_model}
      d_state: 16
      d_conv: 4
      expand: 2
```

### SSM vs Attention

| Aspect | Attention | SSM |
|--------|-----------|-----|
| Complexity | O(T²) | O(T) |
| Long context | Memory-bound | Efficient |
| Parallelism | Fully parallel | Parallel scan |
| Training speed | Fast | Competitive |

---

## Normalization

### RMS Normalization

Root Mean Square normalization (no centering):

```yaml
- type: RMSNormLayer
  d_model: 512
  eps: 1.0e-5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Dimension to normalize |
| `eps` | float | `1e-5` | Numerical stability epsilon |

### Layer Normalization

Standard layer normalization with mean centering:

```yaml
- type: LayerNormLayer
  d_model: 512
  eps: 1.0e-5
```

### When to Use Each

| Norm | Use Case |
|------|----------|
| `RMSNormLayer` | Transformers (Llama, GPT-NeoX) — faster, no centering |
| `LayerNormLayer` | Classic transformers (GPT-2, BERT) |

---

## Embeddings

Embeddings are configured in the model's `embedder` section:

### Token Embeddings

```yaml
model:
  type: TransformerModel
  embedder:
    type: token
    vocab_size: 50257
    d_model: 512
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | required | Vocabulary size |
| `d_model` | int | required | Embedding dimension |

### Patch Embeddings (Vision)

```yaml
model:
  type: TransformerModel
  embedder:
    type: patch
    image_size: 224
    patch_size: 16
    in_channels: 3
    d_model: 768
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_size` | int | required | Input image size |
| `patch_size` | int | required | Patch size |
| `in_channels` | int | `3` | Input channels |
| `d_model` | int | required | Embedding dimension |

---

## LoRA

Low-Rank Adaptation for efficient fine-tuning:

### LoRA Layer

Wraps a linear layer with low-rank adapters:

```yaml
- type: LoRALinearLayer
  d_in: 512
  d_out: 512
  rank: 8
  alpha: 16
  dropout: 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_in` | int | required | Input dimension |
| `d_out` | int | required | Output dimension |
| `rank` | int | required | LoRA rank (lower = fewer params) |
| `alpha` | float | `rank` | Scaling factor |
| `dropout` | float | `0.0` | LoRA dropout |

### LoRA on Attention

```yaml
- type: AttentionLayer
  d_model: ${d_model}
  n_heads: ${n_heads}
  mode: standard
  lora_rank: 8
  lora_alpha: 16
  lora_targets: [q, v]   # Apply LoRA to Q and V projections
```

| LoRA Target | What it adapts |
|-------------|----------------|
| `q` | Query projection |
| `k` | Key projection |
| `v` | Value projection |
| `o` | Output projection |

### LoRA Fine-tuning Preset

See `config/presets/lora_finetune.yml` for a complete example.

---

## Utility Layers

### Linear Layer

Basic linear projection:

```yaml
- type: LinearLayer
  d_in: 512
  d_out: 50257
  bias: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_in` | int | required | Input dimension |
| `d_out` | int | required | Output dimension |
| `bias` | bool | `true` | Include bias |

### Dropout Layer

```yaml
- type: DropoutLayer
  p: 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | float | required | Dropout probability |

---

## Diffusion Head

The `DiffusionNextTokenHead` adds diffusion-based token prediction on top of autoregressive transformers.

### Configuration

```python
layer.diffusion_head import DiffusionNextTokenHead, DiffusionHeadConfig

config = DiffusionHeadConfig(
    enabled=True,
    num_train_timesteps=1000,   # Training diffusion steps
    num_infer_steps=12,         # Inference steps (faster)
    time_embed_dim=128,         # Timestep embedding dimension
    mlp_mult=4,                 # Denoiser MLP multiplier
    cfg_dropout_p=0.10,         # CFG conditioning dropout
    cfg_guidance_scale=1.5,     # Classifier-free guidance scale
    scheduler="ddim",           # Scheduler: ddim, ddpm, dpm
    loss_weight=0.10,           # Weight in combined loss
)

head = DiffusionNextTokenHead(embed_dim=512, cfg=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `False` | Enable diffusion head |
| `num_train_timesteps` | int | 1000 | Training timesteps |
| `num_infer_steps` | int | 12 | Inference steps |
| `time_embed_dim` | int | 128 | Timestep embedding dim |
| `cfg_guidance_scale` | float | 1.5 | CFG scale |
| `scheduler` | str | `"ddim"` | Scheduler type |

### Use Cases

- **Hybrid generation** — Combine autoregressive and diffusion sampling
- **Classifier-free guidance** — Improved generation quality
- **Lightweight adapter** — Train while backbone is frozen

### Requirements

```bash
pip install diffusers
```

---

## Layer Configuration Reference

### All Layer Types

| Layer | Purpose | Key Parameters |
|-------|---------|----------------|
| `AttentionLayer` | Multi-head attention | `d_model`, `n_heads`, `mode` |
| `SwiGLULayer` | SwiGLU feed-forward | `d_model`, `d_ff` |
| `GLULayer` | Configurable GLU | `d_model`, `d_ff`, `activation` |
| `MoELayer` | Mixture of Experts | `d_model`, `num_experts`, `top_k` |
| `SSMLayer` | State Space Model | `d_model`, `d_state` |
| `RMSNormLayer` | RMS normalization | `d_model`, `eps` |
| `LayerNormLayer` | Layer normalization | `d_model`, `eps` |
| `LinearLayer` | Linear projection | `d_in`, `d_out`, `bias` |
| `DropoutLayer` | Dropout | `p` |
| `LoRALinearLayer` | LoRA-wrapped linear | `d_in`, `d_out`, `rank` |
| `DiffusionNextTokenHead` | Diffusion token prediction | `embed_dim`, `cfg` |

---

## Programmatic Usage

While manifests are the primary interface, layers can be used directly:

```python
layer.attention import AttentionLayer
config.layer import AttentionLayerConfig, AttentionMode

config = AttentionLayerConfig(
    d_model=512,
    n_heads=8,
    mode=AttentionMode.DECOUPLED,
    sem_dim=64,
    geo_dim=128,
    rope_enabled=True,
)

layer = AttentionLayer(config)
output = layer(x)  # x: (B, T, 512)
```

---

<div align="center">

**[← Manifests](manifests.md)** · **[Topologies →](topologies.md)**

</div>
