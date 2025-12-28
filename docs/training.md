# 🎓 Training Guide

caramba supports multiple training paradigms, from simple end-to-end training to sophisticated architecture surgery with distillation. This guide covers all training modes and their configurations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Standard Training](#standard-training)
- [Upcycle Training](#upcycle-training)
- [Orchestrated Training](#orchestrated-training)
- [Training Configuration](#training-configuration)
- [Distributed Training](#distributed-training)

---

## Overview

caramba provides three training modes:

| Mode | Trainer | Use Case |
|------|---------|----------|
| **Standard** | `trainer.standard` | Training from scratch or fine-tuning |
| **Upcycle** | `trainer.upcycle` | Architecture surgery + distillation |
| **Orchestrated** | Built into runs | Dynamic optimizer switching |

Each mode is selected by the `trainer` field and configured in `train`:

```yaml
targets:
  - type: experiment
    trainer: trainer.standard  # or trainer.upcycle
    runs:
      - id: train
        train:
          phase: standard
          orchestrator_enabled: false  # Enable for orchestrated mode
```

---

## Standard Training

End-to-end training from scratch or fine-tuning an existing model.

### Basic Configuration

```yaml
trainer: trainer.standard

runs:
  - id: train
    mode: train
    steps: 10000
    train:
      phase: standard
      batch_size: 32
      block_size: 512
      lr: 0.0003
      device: mps
      dtype: float32
```

### Full Options

```yaml
train:
  phase: standard

  # Core settings
  batch_size: 32          # Training batch size
  block_size: 512         # Sequence length
  lr: 0.0003              # Learning rate
  device: mps             # Device: mps, cuda, cpu
  dtype: float32          # Data type: float32, float16, bfloat16

  # Optimizer settings
  weight_decay: 0.01      # Weight decay (AdamW)
  beta1: 0.9              # Adam beta1
  beta2: 0.95             # Adam beta2
  grad_clip: 1.0          # Gradient clipping norm

  # Learning rate schedule
  warmup_steps: 100       # LR warmup steps
  lr_schedule: cosine     # cosine, linear, constant
  min_lr: 0.00001         # Minimum LR for schedule

  # Optimization features
  use_amp: false          # Automatic mixed precision
  amp_dtype: float16      # AMP dtype
  compile_model: false    # torch.compile optimization
  gradient_accumulation_steps: 1

  # Data loading
  num_workers: 4          # DataLoader workers
  pin_memory: true        # Pin memory for GPU transfer
```

### Example: Training a Small Transformer

```yaml
version: 2
name: train_from_scratch

vars:
  d_model: 256
  n_heads: 4
  n_layers: 4

targets:
  - type: experiment
    name: baseline
    trainer: trainer.standard
    runs:
      - id: train
        mode: train
        steps: 5000
        train:
          phase: standard
          batch_size: 16
          block_size: 256
          lr: 0.001
          device: mps
          dtype: float32
          warmup_steps: 200
          lr_schedule: cosine
```

Run it:

```bash
python3 -m caramba my_training.yml
```

---

## Upcycle Training

Architecture surgery that converts a pretrained model to a new architecture while preserving learned representations. This is caramba's flagship feature for attention surgery research.

### The Upcycle Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                    UPCYCLE PIPELINE                       │
├──────────────────────────────────────────────────────────┤
│  1. Load Teacher     Load pretrained checkpoint          │
│  2. Build Student    Create target architecture          │
│  3. Surgery          SVD-based weight initialization     │
│  4. Blockwise        Layer-by-layer distillation         │
│  5. Global           End-to-end fine-tuning              │
│  6. Verify           Compare teacher/student outputs     │
│  7. Benchmark        Measure quality/speed/memory        │
└──────────────────────────────────────────────────────────┘
```

### Upcycle Configuration

```yaml
trainer: trainer.upcycle

runs:
  # Phase 1: Blockwise distillation
  - id: blockwise
    mode: train
    steps: 500
    train:
      phase: blockwise
      teacher_ckpt: hf://meta-llama/Llama-3.2-1B
      batch_size: 1
      block_size: 2048
      lr: 0.0001
      device: mps
      dtype: float32

  # Phase 2: Global fine-tuning
  - id: finetune
    mode: train
    steps: 2000
    train:
      phase: global
      batch_size: 1
      block_size: 2048
      lr: 0.00005
      device: mps
      dtype: float32
```

### Training Phases

#### Blockwise Phase

Trains each layer to match teacher outputs:

```yaml
train:
  phase: blockwise
  teacher_ckpt: hf://meta-llama/Llama-3.2-1B

  # Convergence-based training (optional)
  convergence_target: 0.02      # Target L1 loss
  convergence_patience: 100     # Steps without improvement
  convergence_max_steps: 2000   # Max steps per block

  # Teacher output caching
  cache_teacher_outputs: true   # Cache for speed
```

The blockwise phase:
1. Iterates through each transformer block
2. Runs teacher forward to get target outputs
3. Trains student block to minimize L1 distance
4. Optionally uses convergence-based stopping

#### Global Phase

Fine-tunes the entire model end-to-end:

```yaml
train:
  phase: global
  lr: 0.00005  # Lower LR for fine-tuning
```

The global phase:
1. Unfreezes all parameters
2. Trains on next-token prediction
3. Uses cross-entropy loss

### Llama → DBA Example

Converting Llama 3.2 1B to Decoupled Bottleneck Attention:

```yaml
version: 2
name: llama_to_dba

vars:
  d_model: 2048
  n_heads: 32
  n_kv_heads: 8
  sem_dim: 128    # Semantic bottleneck
  geo_dim: 256    # Geometric bottleneck

targets:
  - type: experiment
    name: upcycle
    trainer: trainer.upcycle

    system:
      ref: system.language_model
      config:
        model:
          type: TransformerModel
          topology:
            type: StackedTopology
            layers:
              - type: NestedTopology
                repeat: 16
                layers:
                  - type: ResidualTopology
                    layers:
                      - type: RMSNormLayer
                        d_model: ${d_model}
                      - type: AttentionLayer
                        d_model: ${d_model}
                        n_heads: ${n_heads}
                        n_kv_heads: ${n_kv_heads}
                        mode: decoupled      # DBA mode
                        sem_dim: ${sem_dim}
                        geo_dim: ${geo_dim}
                        rope_enabled: true
                  # ... FFN blocks ...

    runs:
      - id: blockwise
        train:
          phase: blockwise
          teacher_ckpt: hf://meta-llama/Llama-3.2-1B
          convergence_target: 0.02

      - id: finetune
        train:
          phase: global
          lr: 0.00005
```

### Verification

Attach verification to check quality after training:

```yaml
runs:
  - id: blockwise
    verify:
      type: compare
      batches: 5
      attention:
        max_mean_l1: 0.05
        max_max_l1: 0.25
      logits:
        max_mean_l1: 0.05
        max_max_l1: 0.25
```

See [Manifests → Verification](manifests.md#verification) for all options.

---

## Orchestrated Training

Dynamic optimizer switching based on training telemetry. The orchestrator monitors loss, gradients, and training phase to select the best optimization strategy.

### Why Orchestrated Training?

Different training phases benefit from different strategies:

| Phase | Challenge | Strategy |
|-------|-----------|----------|
| Early | High gradients | Conservative clipping |
| Plateau | Slow progress | Momentum boost |
| Late | Overfitting | SGD for generalization |
| Spike | Loss explosion | Safety rollback |

### Enable Orchestration

```yaml
train:
  phase: global
  orchestrator_enabled: true
  orchestrator_decision_interval: 500
  orchestrator_eval_horizon: 100
  orchestrator_initial_strategy: conservative_adamw
  orchestrator_use_adagc: true
```

### Orchestrator Options

```yaml
train:
  # Enable orchestration
  orchestrator_enabled: true

  # Decision timing
  orchestrator_decision_interval: 500  # Steps between decisions
  orchestrator_eval_horizon: 100       # Steps to evaluate each strategy

  # Initial strategy
  orchestrator_initial_strategy: conservative_adamw

  # Strategy components
  orchestrator_use_adagc: true         # Adaptive gradient clipping
  orchestrator_use_nowcasting: false   # Weight trajectory prediction

  # Safety
  orchestrator_max_loss_increase: 1.5  # Rollback threshold
  orchestrator_safety_strategy: spike_resistant
```

### Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `conservative_adamw` | Safe defaults, moderate LR, global clipping |
| `aggressive_adamw` | Higher LR, less clipping, faster convergence |
| `sgd_escape` | SGD with momentum for escaping sharp minima |
| `spike_resistant` | Low LR, aggressive clipping for unstable phases |

### SWATS: Automatic Adam → SGD

The orchestrator includes SWATS, which automatically switches from Adam to SGD when training stabilizes:

```python
# Programmatic usage
from orchestrator import SWATS, SWATSConfig

optimizer = SWATS(
    model.parameters(),
    config=SWATSConfig(
        adam_lr=1e-3,
        switch_threshold=1e-9,
        min_steps_before_switch=1000,
    ),
)
```

### AdaGC: Adaptive Gradient Clipping

Per-parameter clipping that adapts to each parameter's gradient distribution:

```yaml
train:
  orchestrator_use_adagc: true
  orchestrator_adagc_warmup: 100
  orchestrator_adagc_threshold: 3.0
```

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                            │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Telemetry   │───▶│   Decision   │───▶│   Strategy   │  │
│  │   Stream     │    │   Boundary   │    │   Switch     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │         │
│         ▼                   ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Spike     │    │     UCB      │    │  Speculative │  │
│  │   Detector   │    │    Bandit    │    │   Branching  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## Training Configuration

### Complete Reference

```yaml
train:
  # === Core Settings ===
  phase: standard           # standard, blockwise, global
  batch_size: 32
  block_size: 512
  lr: 0.0003
  device: mps               # mps, cuda, cpu
  dtype: float32            # float32, float16, bfloat16

  # === Upcycle Settings ===
  teacher_ckpt: null        # HF path or local checkpoint
  cache_teacher_outputs: false

  # === Convergence Settings ===
  convergence_target: null  # Target loss for early stopping
  convergence_patience: 100 # Steps without improvement
  convergence_max_steps: null

  # === Optimizer Settings ===
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0

  # === LR Schedule ===
  warmup_steps: 0
  lr_schedule: cosine       # cosine, linear, constant, none
  min_lr: 0.0

  # === Mixed Precision ===
  use_amp: false
  amp_dtype: float16

  # === Optimization ===
  compile_model: false
  gradient_accumulation_steps: 1
  activation_checkpointing: false
  activation_checkpoint_threshold: null

  # === Data Loading ===
  num_workers: 0
  pin_memory: false

  # === Orchestrator ===
  orchestrator_enabled: false
  orchestrator_decision_interval: 500
  orchestrator_eval_horizon: 100
  orchestrator_initial_strategy: conservative_adamw
  orchestrator_use_adagc: false
  orchestrator_use_nowcasting: false
```

### Device Selection

| Device | When to Use |
|--------|-------------|
| `mps` | Apple Silicon (M1/M2/M3/M4) |
| `cuda` | NVIDIA GPUs |
| `cpu` | Development/testing only |

### Data Type Selection

| dtype | Precision | Memory | Speed | Use Case |
|-------|-----------|--------|-------|----------|
| `float32` | Full | High | Baseline | Training stability |
| `float16` | Half | Low | Fast | Inference, AMP |
| `bfloat16` | Brain float | Low | Fast | Training on Ampere+ |

---

## Distributed Training

Scale training to multiple GPUs with DDP or FSDP.

### Data Parallel (DDP)

For models that fit on a single GPU:

```python
from trainer import DistributedConfig, DistributedStrategy

dist_config = DistributedConfig(
    strategy=DistributedStrategy.DDP,
    ddp_find_unused_parameters=False,
)
```

Launch:

```bash
torchrun --nproc_per_node=4 train.py
```

### Fully Sharded (FSDP)

For models that don't fit on a single GPU:

```python
dist_config = DistributedConfig(
    strategy=DistributedStrategy.FSDP,
    fsdp_sharding_strategy="FULL_SHARD",
    fsdp_mixed_precision=True,
    fsdp_activation_checkpointing=True,
    fsdp_transformer_layer_cls=["TransformerBlock"],
)
```

### Distributed Utilities

```python
from trainer.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)

if is_main_process():
    print(f"Training on {get_world_size()} GPUs")
```

---

## Training Presets

### Quick Experiments

Use the `quick` target for fast iteration:

```bash
python3 -m caramba config/presets/llama32_1b_dba.yml --target quick
```

This runs with:
- Reduced steps (50 blockwise, 100 global)
- Smaller block size (512)
- Minimal benchmarks

### Full Paper Runs

Use the `paper` target for publication-quality experiments:

```bash
python3 -m caramba config/presets/llama32_1b_dba.yml --target paper
```

This runs with:
- Full training (500 blockwise, 2000 global)
- Full block size (2048)
- Complete benchmarks (perplexity, latency, memory)
- Artifact generation (CSV, PNG, LaTeX)

---

## Summary

| Mode | Trainer | Phases | Use Case |
|------|---------|--------|----------|
| Standard | `trainer.standard` | `standard` | From-scratch training |
| Upcycle | `trainer.upcycle` | `blockwise` → `global` | Architecture surgery |
| Orchestrated | Any + flags | Any | Adaptive optimization |

---

<div align="center">

**[← Topologies](topologies.md)** · **[Inference →](inference.md)**

</div>
