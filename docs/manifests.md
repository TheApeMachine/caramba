# 📄 Manifest Reference

Manifests are YAML files that declaratively define experiments in caramba. This document covers the complete schema, all options, and patterns for common use cases.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Top-Level Structure](#top-level-structure)
- [Variables](#variables)
- [Defaults](#defaults)
- [Targets](#targets)
- [Runs](#runs)
- [Verification](#verification)
- [Benchmarks](#benchmarks)
- [Entrypoints](#entrypoints)
- [Presets](#presets)

---

## Overview

A manifest is a complete experiment definition that caramba compiles and executes:

```yaml
version: 2
name: my_experiment
notes: What this experiment is about

vars: { ... }       # Reusable variables
defaults: { ... }   # Default settings
targets: [ ... ]    # Runnable units
entrypoints: { ... } # Named entry points (optional)
```

The pipeline processes manifests as:

```text
parse → lower → validate → build → run → verify → benchmark → artifacts
```

---

## Top-Level Structure

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | `int` | Schema version (always `2`) |
| `defaults` | `object` | Default settings for data, logging, runtime |
| `targets` | `list` | One or more experiment/process targets |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Experiment name (used for artifact directories) |
| `notes` | `string` | Human-readable description |
| `vars` | `object` | Variables for `${substitution}` |
| `entrypoints` | `object` | Named targets for `--target` selection |

---

## Variables

Use `vars` to define reusable values throughout your manifest:

```yaml
vars:
  d_model: 2048
  n_heads: 32
  n_layers: 16
  d_ff: 8192
  vocab_size: 128256

  # Training hyperparameters
  lr: 0.0001
  batch_size: 4
  block_size: 2048
```

Reference variables with `${variable_name}`:

```yaml
topology:
  layers:
    - type: AttentionLayer
      d_model: ${d_model}   # Substituted to 2048
      n_heads: ${n_heads}   # Substituted to 32
```

### Variable Types

Variables can be any JSON-compatible type:

```yaml
vars:
  # Numbers
  d_model: 512
  lr: 0.0001

  # Strings
  device: mps

  # Booleans
  use_amp: true

  # Lists (less common, but supported)
  prompt_lengths: [128, 512, 1024]
```

---

## Defaults

The `defaults` section sets experiment-wide defaults for three concerns:

### Data Defaults

```yaml
defaults:
  data:
    tokenizer: llama       # Tokenizer type: llama, tiktoken, gpt2
    val_frac: 0.05         # Validation split fraction
```

### Logging Defaults

```yaml
defaults:
  logging:
    instrument: rich       # Console output: rich, plain
    wandb: true            # Enable Weights & Biases
    wandb_project: my-proj # W&B project name
    wandb_entity: ''       # W&B team/user
    wandb_mode: online     # online, offline, disabled
    eval_iters: 50         # Eval iterations for validation loss
```

### Runtime Defaults

```yaml
defaults:
  runtime:
    save_every: 500        # Checkpoint frequency (steps)
```

---

## Targets

A target is a runnable unit. There are two types:

### Experiment Targets

```yaml
targets:
  - type: experiment
    name: train_baseline
    description: Train a transformer from scratch
    backend: torch
    task: task.language_modeling
    data: { ... }
    system: { ... }
    objective: objective.next_token_ce
    trainer: trainer.standard
    runs: [ ... ]
    benchmarks: [ ... ]
```

#### Experiment Fields

| Field | Required | Description |
|-------|----------|-------------|
| `type` | ✅ | Always `experiment` |
| `name` | ✅ | Unique identifier for this target |
| `description` | ❌ | Human-readable description |
| `backend` | ✅ | Execution backend (`torch`) |
| `task` | ✅ | Task type (see below) |
| `data` | ✅ | Data configuration (ref + config) |
| `system` | ✅ | Model system configuration |
| `objective` | ✅ | Loss function reference |
| `trainer` | ✅ | Trainer reference |
| `runs` | ✅ | List of training runs |
| `benchmarks` | ❌ | List of benchmarks to run |

#### Task Types

| Task | Use Case |
|------|----------|
| `task.language_modeling` | Next-token prediction (transformers) |
| `task.classification` | Classification (MLP, CNN) |
| `task.node_classification` | Graph node classification |
| `task.denoising` | Diffusion denoising |

#### Data Configuration

```yaml
data:
  ref: dataset.tokens
  config:
    path: fineweb_100m.npy
    block_size: 2048
```

#### System Configuration

```yaml
system:
  ref: system.language_model
  config:
    model:
      type: TransformerModel
      embedder: { ... }
      topology: { ... }

#### Graph System (`system.graph`)

`system.graph` runs a named-port DAG over a TensorDict. Unlike the single-stream
topologies used by language models, GraphTopology nodes read from one or more
input keys and write one or more output keys.

The graph topology is declared as a `GraphTopology` with `layers`. In graph
terminology, these layers are often called *nodes*; manifests may use `nodes` as
an alias for `layers`.

```yaml
system:
  ref: system.graph
  config:
    topology:
      type: GraphTopology
      layers:
        # Layer-backed node (inline LayerConfig payload)
        - id: proj
          in: inputs
          out: h
          type: LinearLayer
          d_in: 4
          d_out: 8
          bias: true

        # Op-backed node (torch.nn.* or python:module:Symbol)
        - id: act
          in: h
          out: h2
          op: ReLU
          config: {}
```
```

### Process Targets (Agent Workflows)

```yaml
targets:
  - type: process
    name: paper_write
    team:
      writer: writer
    process:
      type: paper_write
      name: paper_write
      writer: writer
      output_dir: paper
```

See [Agent Workflows](agents.md) for details.

---

## Runs

Each experiment target contains one or more runs that execute sequentially:

```yaml
runs:
  - id: blockwise
    mode: train
    exp: dba_blockwise
    seed: 42
    steps: 500
    expected:
      phase: blockwise
    verify: { ... }
    train: { ... }

  - id: finetune
    mode: train
    exp: dba_finetune
    seed: 42
    steps: 2000
    train: { ... }
```

### Run Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique run identifier |
| `mode` | ✅ | Run mode (`train`) |
| `exp` | ❌ | Experiment name for logging |
| `seed` | ❌ | Random seed |
| `steps` | ✅ | Number of training steps |
| `expected` | ❌ | Expected values for validation |
| `verify` | ❌ | Verification configuration |
| `train` | ✅ | Training configuration |

### Training Configuration

```yaml
train:
  phase: global           # Training phase: standard, blockwise, global
  batch_size: 4           # Batch size
  block_size: 2048        # Sequence length
  lr: 0.0001              # Learning rate
  device: mps             # Device: mps, cuda, cpu
  dtype: float32          # Data type: float32, float16, bfloat16

  # Upcycle-specific (optional)
  teacher_ckpt: hf://meta-llama/Llama-3.2-1B

  # Convergence-based training (optional)
  convergence_target: 0.02
  convergence_patience: 100
  convergence_max_steps: 2000

  # Optimization (optional)
  cache_teacher_outputs: true
  use_amp: false
  amp_dtype: float16
  gradient_accumulation_steps: 1
  num_workers: 0
  pin_memory: false
  compile_model: false

  # Orchestrator (optional)
  orchestrator_enabled: false
  orchestrator_decision_interval: 500
  orchestrator_initial_strategy: conservative_adamw
  orchestrator_use_adagc: true
```

### Training Phases

| Phase | Description | Trainer |
|-------|-------------|---------|
| `standard` | End-to-end training | `trainer.standard` |
| `blockwise` | Layer-by-layer distillation | `trainer.upcycle` |
| `global` | Full model fine-tuning | `trainer.upcycle` |

---

## Verification

Attach verification to runs to validate model behavior after training:

### Compare Verification

Compare teacher and student outputs:

```yaml
verify:
  type: compare
  batches: 5              # Number of batches to compare
  fail_fast: false        # Continue on failure
  attention:
    max_mean_l1: 0.05     # Max mean L1 on attention outputs
    max_max_l1: 0.25      # Max L1 (worst case)
  logits:
    max_mean_l1: 0.05
    max_max_l1: 0.25
```

### Fidelity Verification

Loss-based quality gate:

```yaml
verify:
  type: fidelity
  batches: 5
  split: auto             # auto, train, val
  max_delta_nll: 0.05     # teacher_nll - student_nll
  max_ppl_ratio: 1.05     # student_ppl / teacher_ppl
  fail_fast: false
```

### Eval Verification

Behavioral test cases:

```yaml
verify:
  type: eval
  tokenizer:
    type: llama
  max_new_tokens: 32
  thresholds:
    min_student_accuracy: 0.7
    max_accuracy_drop: 0.1
  cases:
    - id: math_simple
      prompt: "What is 2 + 2? Answer:"
      answer: 4
      kind: int_greedy      # Parse first integer

    - id: capital_france
      prompt: "The capital of France is"
      choices: [Paris, London, Berlin, Madrid]
      answer: Paris
      kind: choice_logprob  # Rank by log probability
```

---

## Benchmarks

Run after training to measure and compare models:

```yaml
benchmarks:
  - id: perplexity
    config:
      type: perplexity
      dataset: fineweb_100m.npy
      block_size: 2048
      batch_size: 1
      num_batches: 100
    models: [teacher, student]
    repeats: 1

  - id: latency
    config:
      type: latency
      prompt_lengths: [128, 512, 1024, 2048]
      generation_lengths: [64, 128, 256]
      batch_sizes: [1]
      warmup_runs: 3
      timed_runs: 10
      use_cache: true
      cache_kind: fp16
    models: [teacher, student]

  - id: memory
    config:
      type: memory
      sequence_lengths: [512, 1024, 2048, 4096]
      batch_sizes: [1]
      measure_peak: true
      measure_kvcache: true
      quantization_modes: [fp16, q8, q4]
    models: [teacher, student]
```

### Benchmark Types

| Type | Measures | Artifacts |
|------|----------|-----------|
| `perplexity` | Cross-entropy loss, PPL | perplexity.csv |
| `latency` | Tokens/sec, prefill/decode time | latency.csv, latency_vs_context.png |
| `memory` | KV-cache size, peak memory | memory.csv, memory_scaling.png |

Benchmarks generate:
- **CSV files** — Raw measurements
- **PNG charts** — Visualization
- **LaTeX tables** — Paper-ready tables

---

## Entrypoints

Define named entry points for convenience:

```yaml
entrypoints:
  default: paper          # Used when --target is not specified
  quick: quick_validation
  full: paper
```

Usage:

```bash
# Uses entrypoints.default (paper)
python3 -m caramba manifest.yml

# Explicit target
python3 -m caramba manifest.yml --target quick
```

---

## Presets

caramba includes ready-to-use presets in `config/presets/`:

### Language Modeling

| Preset | Architecture | Description |
|--------|--------------|-------------|
| `llama32_1b_dba.yml` | Llama 3.2 1B → DBA | Full upcycle with benchmarks |
| `standard_transformer.yml` | GPT-style | Baseline transformer training |
| `moe_transformer.yml` | Transformer + MoE | Mixture of Experts |
| `mamba_ssm.yml` | Mamba | State Space Model |
| `lora_finetune.yml` | LoRA | Efficient fine-tuning |

### Vision

| Preset | Architecture | Description |
|--------|--------------|-------------|
| `vit.yml` | Vision Transformer | Image classification |

### Other

| Preset | Architecture | Description |
|--------|--------------|-------------|
| `mlp_classifier.yml` | MLP | Simple classification |
| `diffusion_vector.yml` | Diffusion | Denoising model |
| `graph_node_classification.yml` | GCN | Graph neural network |

### Using Presets

```bash
# Dry-run to see the plan
python3 -m caramba config/presets/moe_transformer.yml --dry-run

# Run with default target
python3 -m caramba config/presets/llama32_1b_dba.yml

# Run specific target
python3 -m caramba config/presets/llama32_1b_dba.yml --target quick
```

### Customizing Presets

Copy a preset and modify it:

```bash
cp config/presets/standard_transformer.yml my_experiment.yml
# Edit my_experiment.yml with your changes
python3 -m caramba my_experiment.yml
```

---

## Complete Example

Here's a full manifest demonstrating most features:

```yaml
version: 2
name: complete_example
notes: Demonstrates all manifest features

vars:
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  vocab_size: 50257
  block_size: 512
  batch_size: 8

defaults:
  data:
    tokenizer: tiktoken
    val_frac: 0.1
  logging:
    instrument: rich
    wandb: true
    wandb_project: caramba-example
  runtime:
    save_every: 200

targets:
  - type: experiment
    name: full_pipeline
    description: Train, verify, and benchmark
    backend: torch
    task: task.language_modeling

    data:
      ref: dataset.tokens
      config:
        path: data.npy
        block_size: ${block_size}

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
                        mode: standard
                  - type: ResidualTopology
                    layers:
                      - type: RMSNormLayer
                        d_model: ${d_model}
                      - type: SwiGLULayer
                        d_model: ${d_model}
                        d_ff: ${d_ff}
              - type: RMSNormLayer
                d_model: ${d_model}
              - type: LinearLayer
                d_in: ${d_model}
                d_out: ${vocab_size}

    objective: objective.next_token_ce
    trainer: trainer.standard

    runs:
      - id: train
        mode: train
        exp: example_train
        seed: 42
        steps: 1000
        verify:
          type: fidelity
          batches: 5
          max_delta_nll: 0.1
        train:
          phase: standard
          batch_size: ${batch_size}
          block_size: ${block_size}
          lr: 0.0003
          device: mps
          dtype: float32

    benchmarks:
      - id: perplexity
        config:
          type: perplexity
          dataset: data.npy
          block_size: 512
          num_batches: 50
        models: [student]

entrypoints:
  default: full_pipeline
```

---

<div align="center">

**[← Getting Started](getting-started.md)** · **[Layers →](layers.md)**

</div>
