# 🔗 Topology Guide

Topologies define how layers connect and compose. They are the structural building blocks that turn individual layers into complete architectures. This guide covers all topology types and patterns for building complex models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Topology Types](#topology-types)
- [Composition Patterns](#composition-patterns)
- [Common Architectures](#common-architectures)
- [Advanced Patterns](#advanced-patterns)

---

## Overview

Topologies are graph nodes that contain layers or other topologies:

```yaml
topology:
  type: StackedTopology       # Root topology
  layers:
    - type: NestedTopology    # Contains repeated blocks
      repeat: 6
      layers:
        - type: ResidualTopology   # Skip connection
          layers:
            - type: RMSNormLayer   # Actual computation
            - type: AttentionLayer
```

Key principles:
- **Topologies define structure** — How data flows between components
- **Layers define computation** — What happens to the data
- **Topologies are composable** — Nest them arbitrarily
- **Topologies are declarative** — No procedural code needed

---

## Topology Types

### StackedTopology

Sequential execution — output of each layer feeds into the next:

```yaml
topology:
  type: StackedTopology
  layers:
    - type: LayerA
    - type: LayerB
    - type: LayerC
```

**Data flow:** `x → A → B → C → output`

**Use cases:**
- Transformer blocks in sequence
- Any feedforward pipeline
- Combining pre-norm, attention, and output

**Example — Simple feedforward:**

```yaml
topology:
  type: StackedTopology
  layers:
    - type: LinearLayer
      d_in: 784
      d_out: 256
    - type: DropoutLayer
      p: 0.1
    - type: LinearLayer
      d_in: 256
      d_out: 10
```

---

### ResidualTopology

Skip connection that adds input to output: `output = x + f(x)`

```yaml
topology:
  type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: 512
    - type: AttentionLayer
      d_model: 512
      n_heads: 8
```

**Data flow:** `x → [norm → attn] → x + result`

**Use cases:**
- Pre-norm transformer blocks
- ResNet-style connections
- Any layer that benefits from gradient shortcuts

**Example — Pre-norm attention block:**

```yaml
- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: AttentionLayer
      d_model: ${d_model}
      n_heads: ${n_heads}
      mode: standard
```

---

### NestedTopology

Repeats a sequence of layers N times:

```yaml
topology:
  type: NestedTopology
  repeat: 12
  layers:
    - type: ResidualTopology
      layers: [...]
    - type: ResidualTopology
      layers: [...]
```

**Data flow:** Loops through layers `repeat` times, passing output to next iteration

**Use cases:**
- Stacking N identical transformer blocks
- Repeating any architectural pattern
- Reducing manifest verbosity

**Example — 12-layer transformer:**

```yaml
topology:
  type: StackedTopology
  layers:
    - type: NestedTopology
      repeat: 12
      layers:
        # Attention block
        - type: ResidualTopology
          layers:
            - type: RMSNormLayer
              d_model: 512
            - type: AttentionLayer
              d_model: 512
              n_heads: 8
        # FFN block
        - type: ResidualTopology
          layers:
            - type: RMSNormLayer
              d_model: 512
            - type: SwiGLULayer
              d_model: 512
              d_ff: 2048
    # Final norm
    - type: RMSNormLayer
      d_model: 512
```

---

### ParallelTopology

Executes layers in parallel and stacks outputs along a new dimension:

```yaml
topology:
  type: ParallelTopology
  layers:
    - type: LinearLayer
      d_in: 512
      d_out: 64
    - type: LinearLayer
      d_in: 512
      d_out: 64
    - type: LinearLayer
      d_in: 512
      d_out: 64
```

**Data flow:** `x → [A(x), B(x), C(x)] → stack([a, b, c])`

**Use cases:**
- Multi-head projections
- Ensemble-style architectures
- Parallel feature extraction

**Example — Multi-head Q/K/V:**

```yaml
- type: ParallelTopology
  layers:
    - type: LinearLayer
      d_in: 512
      d_out: 512  # Q
    - type: LinearLayer
      d_in: 512
      d_out: 512  # K
    - type: LinearLayer
      d_in: 512
      d_out: 512  # V
```

---

### BranchingTopology

Executes layers in parallel and concatenates outputs:

```yaml
topology:
  type: BranchingTopology
  layers:
    - type: LinearLayer
      d_in: 512
      d_out: 256
    - type: LinearLayer
      d_in: 512
      d_out: 256
```

**Data flow:** `x → [A(x), B(x)] → concat([a, b])`

**Use cases:**
- Feature fusion from multiple paths
- Inception-style modules
- Multi-scale processing

**Example — Dual-path feature extraction:**

```yaml
- type: BranchingTopology
  layers:
    # Path 1: Local features
    - type: StackedTopology
      layers:
        - type: LinearLayer
          d_in: 512
          d_out: 256
    # Path 2: Global features
    - type: StackedTopology
      layers:
        - type: AttentionLayer
          d_model: 512
          n_heads: 8
        - type: LinearLayer
          d_in: 512
          d_out: 256
# Output dimension: 256 + 256 = 512
```

---

### CyclicTopology

Creates cyclic connections for iterative refinement:

```yaml
topology:
  type: CyclicTopology
  iterations: 3
  layers:
    - type: AttentionLayer
      d_model: 512
      n_heads: 8
```

**Data flow:** Repeats layers `iterations` times with feedback

**Use cases:**
- Iterative refinement models
- Graph neural networks with message passing
- Recurrent-like processing without explicit state

---

### RecurrentTopology

Recurrent execution with cache passthrough for stateful processing:

```yaml
topology:
  type: RecurrentTopology
  layers:
    - type: SSMLayer
      d_model: 512
      d_state: 16
```

**Data flow:** Maintains state across forward calls

**Use cases:**
- SSM-based models (Mamba)
- RNN-style architectures
- Streaming inference

---

### GraphTopology

Named-port DAG execution over a `TensorDict` / `dict[str, Tensor]`.

Instead of a single tensor stream (`x → layer → layer → ...`), a GraphTopology reads and writes **named keys**. Each node declares:

- `id`: unique node id
- `op`: a Caramba `LayerType` (e.g. `Conv2dLayer`, `DenseLayer`), a `torch.nn` module name (e.g. `ReLU`, `MaxPool2d`), or a Python symbol via `python:module:Symbol`
- `in` / `out`: input/output keys (string or list of strings)
- `config`: kwargs passed to the op constructor
- `repeat` (optional): repeats a single-in/single-out node with chained keys

GraphTopology can also declare an optional input contract:

- `inputs`: list of keys that must be present in the input batch (compile-time validation)

**Data flow:** keys, not positions. Example: `batch["x"] → node(conv1) → batch["h1"] → ...`.

#### Example — Vision CNN (Conv2d → Pool → Dense)

```yaml
system:
  ref: system.generic
  config:
    model:
      type: MLPModel
      topology:
        type: GraphTopology
        inputs: [x]
        nodes:
          - id: conv1
            op: Conv2dLayer
            in: x
            out: h1
            config:
              in_channels: 3
              out_channels: 16
              kernel_size: 3
              padding: 1

          - id: relu1
            op: ReLU
            in: h1
            out: h2
            config: {}

          - id: pool1
            op: MaxPool2d
            in: h2
            out: h3
            config:
              kernel_size: 2
              stride: 2

          - id: flatten
            op: Flatten
            in: h3
            out: h4
            config:
              start_dim: 1

          - id: head
            op: DenseLayer
            in: h4
            out: logits
            config:
              d_in: 3136      # 16 * 14 * 14 (if input is 28x28)
              d_out: 10
              activation: null
              normalization: null
              dropout: 0.0
```

#### Example — Graph node classification (GCN)

```yaml
system:
  ref: system.generic
  config:
    model:
      type: MLPModel
      topology:
        type: GraphTopology
        inputs: [x, adj]
        nodes:
          - id: gcn1
            op: GraphConvLayer
            in: [x, adj]
            out: h
            config:
              kind: gcn
              in_features: 16
              out_features: 32
              bias: true

          - id: relu
            op: ReLU
            in: h
            out: h2
            config: {}

          - id: head
            op: DenseLayer
            in: h2
            out: logits
            config:
              d_in: 32
              d_out: 7
              activation: null
              normalization: null
              dropout: 0.0
```

#### Example — Hybrid (two encoders + concat + head)

```yaml
system:
  ref: system.generic
  config:
    model:
      type: MLPModel
      topology:
        type: GraphTopology
        inputs: [text_emb, image_emb]
        nodes:
          - id: text_enc
            op: DenseLayer
            in: text_emb
            out: t
            config:
              d_in: 768
              d_out: 256
              activation: gelu

          - id: image_enc
            op: DenseLayer
            in: image_emb
            out: v
            config:
              d_in: 1024
              d_out: 256
              activation: gelu

          - id: fuse
            op: python:caramba.topology.ops:Concat
            in: [t, v]
            out: fused
            config:
              dim: -1

          - id: head
            op: DenseLayer
            in: fused
            out: logits
            config:
              d_in: 512
              d_out: 5
              activation: null
```

---

## Composition Patterns

### Pattern 1: Standard Transformer Block

The most common pattern — attention + FFN with residuals:

```yaml
# One transformer block
- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: AttentionLayer
      d_model: ${d_model}
      n_heads: ${n_heads}

- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: SwiGLULayer
      d_model: ${d_model}
      d_ff: ${d_ff}
```

### Pattern 2: Full Transformer

Embedding + N blocks + output:

```yaml
model:
  type: TransformerModel
  embedder:
    type: token
    vocab_size: ${vocab_size}
    d_model: ${d_model}
  topology:
    type: StackedTopology
    layers:
      # N transformer blocks
      - type: NestedTopology
        repeat: ${n_layers}
        layers:
          - type: ResidualTopology
            layers:
              - type: RMSNormLayer
                d_model: ${d_model}
              - type: AttentionLayer
                d_model: ${d_model}
                n_heads: ${n_heads}
          - type: ResidualTopology
            layers:
              - type: RMSNormLayer
                d_model: ${d_model}
              - type: SwiGLULayer
                d_model: ${d_model}
                d_ff: ${d_ff}
      # Final norm
      - type: RMSNormLayer
        d_model: ${d_model}
      # LM head
      - type: LinearLayer
        d_in: ${d_model}
        d_out: ${vocab_size}
```

### Pattern 3: MoE Transformer

Replace FFN with Mixture of Experts:

```yaml
- type: NestedTopology
  repeat: ${n_layers}
  layers:
    # Attention
    - type: ResidualTopology
      layers:
        - type: RMSNormLayer
          d_model: ${d_model}
        - type: AttentionLayer
          d_model: ${d_model}
          n_heads: ${n_heads}
    # MoE FFN
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

### Pattern 4: Hybrid SSM-Attention

Alternate between attention and SSM layers:

```yaml
- type: NestedTopology
  repeat: ${n_layers}
  layers:
    # SSM block (odd layers)
    - type: ResidualTopology
      layers:
        - type: RMSNormLayer
          d_model: ${d_model}
        - type: SSMLayer
          d_model: ${d_model}
          d_state: 16
    # Attention block (every N layers)
    - type: ResidualTopology
      layers:
        - type: RMSNormLayer
          d_model: ${d_model}
        - type: AttentionLayer
          d_model: ${d_model}
          n_heads: ${n_heads}
```

---

## Common Architectures

### GPT-style Transformer

```yaml
vars:
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072
  vocab_size: 50257

model:
  type: TransformerModel
  embedder:
    type: token
    vocab_size: ${vocab_size}
    d_model: ${d_model}
  topology:
    type: StackedTopology
    layers:
      - type: NestedTopology
        repeat: ${n_layers}
        layers:
          - type: ResidualTopology
            layers:
              - type: LayerNormLayer
                d_model: ${d_model}
              - type: AttentionLayer
                d_model: ${d_model}
                n_heads: ${n_heads}
                mode: standard
          - type: ResidualTopology
            layers:
              - type: LayerNormLayer
                d_model: ${d_model}
              - type: SwiGLULayer
                d_model: ${d_model}
                d_ff: ${d_ff}
      - type: LayerNormLayer
        d_model: ${d_model}
      - type: LinearLayer
        d_in: ${d_model}
        d_out: ${vocab_size}
```

### Llama-style (Pre-norm, RMSNorm, RoPE)

```yaml
vars:
  d_model: 2048
  n_heads: 32
  n_kv_heads: 8
  n_layers: 16
  d_ff: 8192
  vocab_size: 128256
  rope_theta: 500000.0

model:
  type: TransformerModel
  embedder:
    type: token
    vocab_size: ${vocab_size}
    d_model: ${d_model}
  topology:
    type: StackedTopology
    layers:
      - type: NestedTopology
        repeat: ${n_layers}
        layers:
          - type: ResidualTopology
            layers:
              - type: RMSNormLayer
                d_model: ${d_model}
                eps: 1e-5
              - type: AttentionLayer
                d_model: ${d_model}
                n_heads: ${n_heads}
                n_kv_heads: ${n_kv_heads}
                mode: gqa
                rope_enabled: true
                rope_base: ${rope_theta}
                is_causal: true
          - type: ResidualTopology
            layers:
              - type: RMSNormLayer
                d_model: ${d_model}
                eps: 1e-5
              - type: SwiGLULayer
                d_model: ${d_model}
                d_ff: ${d_ff}
                bias: false
      - type: RMSNormLayer
        d_model: ${d_model}
        eps: 1e-5
      - type: LinearLayer
        d_in: ${d_model}
        d_out: ${vocab_size}
        bias: false
```

### Vision Transformer (ViT)

```yaml
vars:
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072
  image_size: 224
  patch_size: 16
  num_classes: 1000

model:
  type: TransformerModel
  embedder:
    type: patch
    image_size: ${image_size}
    patch_size: ${patch_size}
    in_channels: 3
    d_model: ${d_model}
  topology:
    type: StackedTopology
    layers:
      - type: NestedTopology
        repeat: ${n_layers}
        layers:
          - type: ResidualTopology
            layers:
              - type: LayerNormLayer
                d_model: ${d_model}
              - type: AttentionLayer
                d_model: ${d_model}
                n_heads: ${n_heads}
                mode: standard
          - type: ResidualTopology
            layers:
              - type: LayerNormLayer
                d_model: ${d_model}
              - type: SwiGLULayer
                d_model: ${d_model}
                d_ff: ${d_ff}
      - type: LayerNormLayer
        d_model: ${d_model}
      # Classification head (on CLS token)
      - type: LinearLayer
        d_in: ${d_model}
        d_out: ${num_classes}
```

### Mamba-style SSM

```yaml
vars:
  d_model: 768
  n_layers: 24
  d_state: 16
  d_conv: 4
  expand: 2

model:
  type: TransformerModel
  embedder:
    type: token
    vocab_size: ${vocab_size}
    d_model: ${d_model}
  topology:
    type: StackedTopology
    layers:
      - type: NestedTopology
        repeat: ${n_layers}
        layers:
          - type: ResidualTopology
            layers:
              - type: RMSNormLayer
                d_model: ${d_model}
              - type: SSMLayer
                d_model: ${d_model}
                d_state: ${d_state}
                d_conv: ${d_conv}
                expand: ${expand}
      - type: RMSNormLayer
        d_model: ${d_model}
      - type: LinearLayer
        d_in: ${d_model}
        d_out: ${vocab_size}
```

---

## Advanced Patterns

### Parallel Attention Heads

Implementing attention as parallel projections:

```yaml
- type: ResidualTopology
  layers:
    - type: RMSNormLayer
      d_model: ${d_model}
    - type: StackedTopology
      layers:
        # Q, K, V projections in parallel
        - type: ParallelTopology
          layers:
            - type: LinearLayer
              d_in: ${d_model}
              d_out: ${d_model}
            - type: LinearLayer
              d_in: ${d_model}
              d_out: ${d_model}
            - type: LinearLayer
              d_in: ${d_model}
              d_out: ${d_model}
        # Custom attention computation would go here
```

### Multi-Scale Processing

Process at multiple resolutions:

```yaml
- type: BranchingTopology
  layers:
    # Fine-grained path
    - type: AttentionLayer
      d_model: 512
      n_heads: 8
    # Coarse path (pooled)
    - type: StackedTopology
      layers:
        - type: LinearLayer
          d_in: 512
          d_out: 128
        - type: AttentionLayer
          d_model: 128
          n_heads: 4
        - type: LinearLayer
          d_in: 128
          d_out: 512
```

### Conditional Computation

Route different tokens to different paths (conceptual):

```yaml
# Note: This is illustrative of the pattern;
# actual conditional routing uses MoELayer
- type: BranchingTopology
  layers:
    - type: SwiGLULayer
      d_model: ${d_model}
      d_ff: ${d_ff}
    - type: AttentionLayer
      d_model: ${d_model}
      n_heads: ${n_heads}
```

---

## Summary

| Topology | Operation | Data Flow |
|----------|-----------|-----------|
| `StackedTopology` | Sequential | A → B → C |
| `ResidualTopology` | Skip connection | x + f(x) |
| `NestedTopology` | Repeat N times | Loop N |
| `ParallelTopology` | Stack outputs | [A, B, C] |
| `BranchingTopology` | Concatenate | cat(A, B) |
| `CyclicTopology` | Iterative | Loop with feedback |
| `RecurrentTopology` | Stateful | Cache passthrough |

---

<div align="center">

**[← Layers](layers.md)** · **[Training →](training.md)**

</div>
