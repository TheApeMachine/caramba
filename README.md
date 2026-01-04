![caramba overview](assets/inside-caramba.png)

# caramba 🧪

**A substrate for architecture research and ML experimentation**

> Architectures are graphs. Graphs are manifests. Running experiments should require nothing more than a YAML file.

caramba provides a frictionless research environment with explicit building blocks, strict validation, and optimized execution. Define your model architecture in YAML, and caramba handles the rest, from compilation to training to publication-ready benchmarks.

---

## 📋 Table of Contents

- [What is caramba?](#-what-is-caramba)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [The Pipeline](#-the-pipeline)
- [Documentation](#-documentation)
- [Available Presets](#-available-presets)
- [Platform Support](#-platform-support)
- [Architecture Overview](#-architecture-overview)

---

## 🎯 What is caramba?

caramba is a declarative ML experimentation platform that separates **intent** from **implementation**:

1. **You declare** what you want in a YAML manifest (architecture, training, benchmarks)
2. **caramba handles** the how (compilation, optimization, execution, artifacts)

This design enables:
- 🔬 **Rapid prototyping** — Test new architectures without writing training loops
- 📊 **Reproducible research** — Manifests are version-controllable experiment definitions
- ⚡ **Automatic optimization** — Runtime planning for batch sizes, precision, and compilation
- 📝 **Publication-ready artifacts** — CSV, PNG, and LaTeX outputs from benchmarks

---

## ✨ Key Features

### 🧱 Generic Layer Library

Built-in support for modern neural network components:

| Layer Type         | Description                                         | Documentation                                 |
|--------------------|-----------------------------------------------------|-----------------------------------------------|
| **Attention**      | Standard, GQA, and DBA (Decoupled Bottleneck) modes | [→ Layers](docs/layers.md#attention)          |
| **MoE**            | Mixture of Experts with load balancing              | [→ Layers](docs/layers.md#mixture-of-experts) |
| **SSM**            | Selective State Space Models (Mamba-style)          | [→ Layers](docs/layers.md#state-space-models) |
| **GLU Variants**   | SwiGLU, GEGLU, and other gated linear units         | [→ Layers](docs/layers.md#feed-forward)       |
| **LoRA**           | Low-rank adaptation for efficient fine-tuning       | [→ Layers](docs/layers.md#lora)               |
| **Normalization**  | RMSNorm, LayerNorm                                  | [→ Layers](docs/layers.md#normalization)      |
| **RoPE**           | Rotary Position Embeddings                          | [→ Layers](docs/layers.md#attention)          |
| **Linear**         | Linear projections with optional bias               | [→ Layers](docs/layers.md#utility-layers)     |
| **Dropout**        | Dropout regularization                              | [→ Layers](docs/layers.md#utility-layers)     |
| **Diffusion Head** | Denoising head for diffusion models                 | [→ Layers](docs/layers.md#utility-layers)     |

### 🔗 Composable Topologies

Define complex model structures declaratively:

| Topology            | Use Case                      | Example              |
|---------------------|-------------------------------|----------------------|
| `StackedTopology`   | Sequential layer execution    | Transformer blocks   |
| `ResidualTopology`  | Skip connections (`x + f(x)`) | Pre-norm blocks      |
| `NestedTopology`    | Repeat layers N times         | N transformer layers |
| `ParallelTopology`  | Execute and stack outputs     | Multi-head attention |
| `BranchingTopology` | Execute and concatenate       | Feature fusion       |
| `CyclicTopology`    | Cyclic connections            | Graph networks       |
| `RecurrentTopology` | Recurrent with cache          | Sequence models      |

[→ Full Topology Guide](docs/topologies.md)

### 🎓 Multiple Training Modes

| Mode             | Description                         | When to Use                  |
|------------------|-------------------------------------|------------------------------|
| **Standard**     | End-to-end training from scratch    | Baseline experiments         |
| **Upcycle**      | Architecture surgery + distillation | Converting pretrained models |
| **Orchestrated** | Dynamic optimizer switching         | Adaptive training research   |

[→ Training Guide](docs/training.md)

### ⚡ Self-Optimization

caramba automatically optimizes your experiments:

- **Runtime planning** — Cached decisions for dtype, AMP, batch size, and `torch.compile`
- **KV-cache policy selection** — Budget-aware quantization with quality gates
- **Decode-plan bucketing** — Dynamic chunking for long-context inference
- **Adaptive speculative decoding** — Auto-adjusting draft lengths

[→ Optimization Details](docs/optimization.md)

### AI Research Collaborators

```bash
python3 -m caramba config/presets/multiplex_chat.yml --target brainstorm
```

The above command puts you in a chat session with ChatGPT 5.2, Claude Opus 4.1, and Gemini Pro 3, which all have the tools they need to inspect the code, perform research, and other relevant actions so you can collaborate on whatever research goals you have.

The agents are not just talking directly with you, but also have the ability to respond to each other, so it should really feel like a team structure.

### 🤖 AI Research Automation

Optional AI-assisted workflows:

- **Paper drafting** — Generate LaTeX documents from experiment results
- **Automated review** — Get reviewer feedback and improvement suggestions
- **Research loop** — Write → Review → Experiment → Repeat

[→ Agent Workflows](docs/agents.md)

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/theapemachine/caramba.git
cd caramba
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Dry-run to validate a manifest (no execution)
python3 -m caramba config/presets/standard_transformer.yml --dry-run

# Run a full experiment with benchmarks
python3 -m caramba config/presets/llama32_1b_dba.yml --target paper

# Quick validation (reduced steps)
python3 -m caramba config/presets/llama32_1b_dba.yml --target quick
```

### Non-LM Architectures

```bash
# MLP classifier
python3 -m caramba config/presets/mlp_classifier.yml --dry-run

# Diffusion model
python3 -m caramba config/presets/diffusion_vector.yml --dry-run

# Graph neural network
python3 -m caramba config/presets/graph_node_classification.yml --dry-run
```

[→ Complete Getting Started Guide](docs/getting-started.md)

---

## 🔄 The Pipeline

Every experiment flows through this chain:

```text
manifest → parse → lower → validate → build → run → verify → benchmark → artifacts
```

| Stage         | What Happens                              |
|---------------|-------------------------------------------|
| **parse**     | Load YAML/JSON, substitute `${variables}` |
| **lower**     | Normalize type names, resolve references  |
| **validate**  | Check schema, verify dimensions           |
| **build**     | Construct PyTorch modules from topology   |
| **run**       | Execute training runs with checkpointing  |
| **verify**    | Compare outputs against thresholds        |
| **benchmark** | Measure perplexity, latency, memory       |
| **artifacts** | Generate CSV, PNG, LaTeX outputs          |

---

## 📚 Documentation

| Guide                                         | Description                                    |
|-----------------------------------------------|------------------------------------------------|
| [🚀 Getting Started](docs/getting-started.md) | Installation, first experiment, basic concepts |
| [📄 Manifest Reference](docs/manifests.md)    | Complete YAML schema with examples             |
| [🧱 Layer Reference](docs/layers.md)          | All layer types and their configurations       |
| [🔗 Topology Guide](docs/topologies.md)       | Composing complex architectures                |
| [🎓 Training Guide](docs/training.md)         | Standard, upcycle, and orchestrated training   |
| [🔮 Inference Guide](docs/inference.md)       | Generation, caching, speculative decoding      |
| [📊 Benchmarking](docs/benchmarking.md)       | Running benchmarks and generating artifacts    |
| [🤖 Agent Workflows](docs/agents.md)          | AI-assisted paper drafting and review          |
| [⚡ Optimization](docs/optimization.md)       | Metal/Triton kernels, runtime planning          |

---

## 📦 Available Presets

Ready-to-use configurations in `config/presets/`:

| Preset | Architecture | Use Case |
|--------|--------------|----------|
| `llama32_1b_dba.yml` | Llama 3.2 1B → DBA | KV-cache compression research |
| `standard_transformer.yml` | GPT-style transformer | Baseline language modeling |
| `moe_transformer.yml` | Transformer + MoE | Sparse scaling research |
| `mamba_ssm.yml` | Mamba-style SSM | Linear-time sequence modeling |
| `vit.yml` | Vision Transformer | Image classification |
| `lora_finetune.yml` | LoRA-enabled model | Efficient fine-tuning |
| `mlp_classifier.yml` | Simple MLP | Non-LM classification |
| `diffusion_vector.yml` | Diffusion denoiser | Generative modeling |
| `graph_node_classification.yml` | GCN | Graph learning |

[→ See all presets with full configurations](docs/manifests.md#presets)

---

## 🖥️ Platform Support

### Apple Silicon (MPS)

caramba treats Apple Silicon as a first-class research target:

- ✅ **Works out of the box** — No special configuration needed
- ✅ **Unified memory** — Fit larger models than discrete GPU VRAM
- ✅ **Metal kernels** — Fused DBA decode for fp16 KV-caches
- ⚠️ **Bandwidth limited** — Expect lower throughput than A100

### NVIDIA CUDA

For maximum throughput:

- ✅ **Triton kernels** — Fused attention decode with quantized caches
- ✅ **DDP/FSDP** — Multi-GPU training support
- ✅ **torch.compile** — Automatic graph optimization

### CPU

Fallback for development and testing:

- ✅ **Full functionality** — All features work
- ⚠️ **Slow** — Not recommended for serious training

---

## 🏗️ Architecture Overview

```text
caramba/
├── config/          # Typed config models, presets, manifests
├── compiler/        # Manifest → executable plan
├── topology/        # Graph nodes (stacked, residual, parallel, ...)
├── layer/           # Thin PyTorch modules (attention, MoE, SSM, ...)
├── model/           # Model building, embedders, trace utilities
├── trainer/         # Training modes (standard, upcycle, orchestrated)
├── infer/           # Generation loop with KV-cache management
├── cache/           # KV-cache with quantization support
├── benchmark/       # Perplexity, latency, memory measurement
├── experiment/      # Unified pipeline orchestration
├── orchestrator/    # Dynamic optimizer switching (SWATS, PIDAO, ...)
├── optimizer/       # Triton (CUDA) + Metal (MPS) fused kernels
├── agent/           # AI research automation (paper, review, loop)
├── instrumentation/ # JSONL/HDF5/TensorBoard/W&B logging
└── console/         # Rich-based logging and progress bars
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest -q

# Run with coverage
coverage run -m pytest && coverage report -m
```

---

## 📄 License

[MIT License](LICENSE.md)

---

<div align="center">

**[Getting Started](docs/getting-started.md)** · **[Manifests](docs/manifests.md)** · **[Layers](docs/layers.md)** · **[Training](docs/training.md)** · **[Inference](docs/inference.md)**

</div>
