# 🚀 Getting Started with caramba

This guide walks you through installing caramba, running your first experiment, and understanding the core concepts that make the platform work.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Your First Experiment](#your-first-experiment)
- [Understanding Manifests](#understanding-manifests)
- [Core Concepts](#core-concepts)
- [Next Steps](#next-steps)

---

## Prerequisites

### Required

- **Python 3.10+** — caramba uses modern Python features
- **PyTorch 2.0+** — For model building and training
- **8GB+ RAM** — For loading models and datasets

### Optional

- **HuggingFace account** — For gated models like Llama (requires `huggingface-cli login`)
- **Xcode Command Line Tools** — For Metal kernel compilation on macOS
- **CUDA + Triton** — For GPU acceleration on NVIDIA hardware

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/theapemachine/caramba.git
cd caramba

# Install dependencies
pip install -r requirements.txt
```

### With Agent Workflows (Optional)

If you want AI-assisted paper drafting and review:

```bash
pip install -e ".[agents]"

# Or install individual components
pip install deeplake docling transformers  # Knowledge store
pip install crawl4ai                        # Web crawling
```

### Verify Installation

```bash
# Should print the execution plan without running
python3 -m caramba config/presets/standard_transformer.yml --dry-run
```

---

## Your First Experiment

Let's run a simple transformer training experiment to verify everything works.

### Step 1: Prepare Data

caramba uses pre-tokenized `.npy` files for efficient data loading. For testing, you can create a small dummy dataset:

```python
import numpy as np

# Create 1M random tokens (replace with real tokenized data for actual experiments)
tokens = np.random.randint(0, 50257, size=1_000_000, dtype=np.int32)
np.save("test_data.npy", tokens)
```

For real experiments, use the FineWeb preparation script:

```bash
python3 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy
```

### Step 2: Create a Manifest

Create `my_experiment.yml`:

```yaml
version: 2
name: my_first_experiment
notes: Learning how caramba works

# Default settings applied to all targets
defaults:
  data:
    tokenizer: tiktoken
    val_frac: 0.1
  logging:
    instrument: rich
    wandb: false
  runtime:
    save_every: 100

# Variables for easy modification
vars:
  d_model: 256
  n_heads: 4
  n_layers: 4
  d_ff: 1024
  vocab_size: 50257
  block_size: 128

# Experiment targets
targets:
  - type: experiment
    name: train
    description: Train a small transformer from scratch
    backend: torch
    task: task.language_modeling

    # Data configuration
    data:
      ref: dataset.tokens
      config:
        path: test_data.npy
        block_size: ${block_size}

    # Model configuration
    system:
      ref: system.language_model
      config:
        model:
          type: TransformerModel
          embedder:
            type: token
            vocab_size: ${vocab_size}
            d_model: ${d_model}
          topology:
            type: StackedTopology
            layers:
              # Repeated transformer blocks
              - type: NestedTopology
                repeat: ${n_layers}
                layers:
                  # Attention with residual
                  - type: ResidualTopology
                    layers:
                      - type: RMSNormLayer
                        d_model: ${d_model}
                      - type: AttentionLayer
                        d_model: ${d_model}
                        n_heads: ${n_heads}
                        mode: standard
                  # FFN with residual
                  - type: ResidualTopology
                    layers:
                      - type: RMSNormLayer
                        d_model: ${d_model}
                      - type: SwiGLULayer
                        d_model: ${d_model}
                        d_ff: ${d_ff}
              # Final normalization
              - type: RMSNormLayer
                d_model: ${d_model}
              # Output projection
              - type: LinearLayer
                d_in: ${d_model}
                d_out: ${vocab_size}

    objective: objective.next_token_ce
    trainer: trainer.standard

    # Training runs
    runs:
      - id: train_small
        mode: train
        exp: my_first_run
        seed: 42
        steps: 500
        train:
          phase: standard
          batch_size: 8
          block_size: ${block_size}
          lr: 0.001
          device: mps  # or 'cuda' or 'cpu'
          dtype: float32
```

### Step 3: Validate the Manifest

Before running, verify the manifest is valid:

```bash
python3 -m caramba my_experiment.yml --dry-run
```

This shows the execution plan without running anything:

```text
┌─────────────────────────────────────────────────────────┐
│ Execution Plan                                          │
├─────────────────────────────────────────────────────────┤
│ Target: train                                           │
│ Runs:                                                   │
│   - train_small (500 steps, device=mps, dtype=float32)  │
│ Benchmarks: []                                          │
└─────────────────────────────────────────────────────────┘
```

### Step 4: Run the Experiment

```bash
python3 -m caramba my_experiment.yml
```

You'll see:

```text
╭─ Training Phase: standard ─╮
│ Step    100/500  loss=5.234 │
│ Step    200/500  loss=4.102 │
│ Step    300/500  loss=3.567 │
│ Step    400/500  loss=3.221 │
│ Step    500/500  loss=2.987 │
╰────────────────────────────╯
✓ Training complete
```

---

## Understanding Manifests

A manifest is a YAML file that declaratively defines your experiment. Here's the structure:

### Top-Level Sections

```yaml
version: 2              # Manifest schema version (always 2)
name: experiment_name   # Used for artifact directories
notes: "Description"    # Human-readable notes

vars:                   # Reusable variables
  d_model: 512

defaults:               # Settings applied to all targets
  data: { ... }
  logging: { ... }
  runtime: { ... }

targets:                # Runnable units (experiments or processes)
  - type: experiment
    name: train
    ...

entrypoints:            # Optional named entry points
  default: "train"
```

### Variable Substitution

Use `${variable}` to reference values from the `vars` section:

```yaml
vars:
  d_model: 512
  n_heads: 8

targets:
  - type: experiment
    system:
      config:
        model:
          topology:
            layers:
              - type: AttentionLayer
                d_model: ${d_model}  # Becomes 512
                n_heads: ${n_heads}  # Becomes 8
```

### The Topology Tree

Models are defined as trees of topologies containing layers:

```yaml
topology:
  type: StackedTopology           # Root: sequential execution
  layers:
    - type: NestedTopology        # Repeat this block N times
      repeat: 6
      layers:
        - type: ResidualTopology  # x + f(x)
          layers:
            - type: RMSNormLayer
            - type: AttentionLayer
```

[→ Full Manifest Reference](manifests.md)

---

## Core Concepts

### 🎯 Targets

A target is a runnable unit. There are two types:

| Type | Purpose |
|------|---------|
| `experiment` | ML training/evaluation with runs and benchmarks |
| `process` | Agent workflow (paper writing, review, etc.) |

### 🔄 Runs

Each experiment target contains one or more runs:

```yaml
runs:
  - id: blockwise
    mode: train
    steps: 500
    train:
      phase: blockwise
      lr: 0.0001

  - id: finetune
    mode: train
    steps: 2000
    train:
      phase: global
      lr: 0.00005
```

Runs execute sequentially within a target.

### 📐 Topologies vs Layers

**Topologies** define structure (how things connect):
- `StackedTopology` — A then B then C
- `ResidualTopology` — x + f(x)
- `ParallelTopology` — [A(x), B(x)] stacked

**Layers** define computation (what happens):
- `AttentionLayer` — Multi-head attention
- `SwiGLULayer` — Feed-forward network
- `RMSNormLayer` — Normalization

### ✅ Verification

Attach verification to runs to check model behavior:

```yaml
runs:
  - id: train
    verify:
      type: compare
      batches: 5
      attention:
        max_mean_l1: 0.05
```

Verification types:
- `compare` — Check L1 distance between teacher/student
- `fidelity` — Check NLL/perplexity ratios
- `eval` — Run behavioral test cases

### 📊 Benchmarks

Measure and compare models after training:

```yaml
benchmarks:
  - id: perplexity
    config:
      type: perplexity
      num_batches: 100
    models: [teacher, student]
```

Generates CSV, PNG, and LaTeX artifacts.

---

## Next Steps

Now that you understand the basics:

1. **[Manifest Reference](manifests.md)** — Complete YAML schema and options
2. **[Layer Reference](layers.md)** — All layer types with configurations
3. **[Topology Guide](topologies.md)** — Building complex architectures
4. **[Training Guide](training.md)** — Standard, upcycle, and orchestrated modes

### Example Experiments to Try

```bash
# Train a Mixture of Experts model
python3 -m caramba config/presets/moe_transformer.yml --dry-run

# Upcycle Llama to DBA (requires HF login)
huggingface-cli login
python3 -m caramba config/presets/llama32_1b_dba.yml --target quick

# Run with full benchmarks
python3 -m caramba config/presets/llama32_1b_dba.yml --target paper
```

---

<div align="center">

**[← Back to README](../README.md)** · **[Manifests →](manifests.md)**

</div>
