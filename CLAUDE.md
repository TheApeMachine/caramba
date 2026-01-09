# CLAUDE.md - Caramba Project Guide

**Last Updated:** 2026-01-09
**Project:** Caramba - Declarative ML Experimentation Platform
**Version:** 2.0

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Key Technologies](#key-technologies)
4. [Directory Structure](#directory-structure)
5. [Getting Started](#getting-started)
6. [Core Concepts](#core-concepts)
7. [Development Guidelines](#development-guidelines)
8. [AI Agent System](#ai-agent-system)
9. [Testing & Quality](#testing--quality)
10. [Deployment](#deployment)
11. [Common Tasks](#common-tasks)

---

## Project Overview

**Caramba** is a declarative ML experimentation platform built for advanced neural network architecture research. The core philosophy is **"Separation of Intent from Implementation"** - researchers define *what* they want to explore in YAML manifests, and Caramba handles the *how* (compilation, optimization, execution, artifacts).

### Mission

- Enable rapid prototyping of novel architectures without writing training loops
- Produce reproducible research with version-controllable experiment definitions
- Provide automatic optimization and publication-ready benchmarks
- Support collaborative AI agents for architecture research discussions

### Key Statistics

- **Lines of Code:** ~800,000+ Python
- **Core Packages:** 46 subpackages
- **Layer Types:** ~35 (attention, SSM, MoE, GLU, LoRA, etc.)
- **Topology Types:** 7
- **Preset Configurations:** 49 ready-to-run experiments
- **Python Version:** 3.12+

---

## Architecture

### Design Philosophy

Caramba follows a **pipeline-based architecture** where YAML manifests flow through several stages:

```
YAML Manifest
    ↓
Parse (substitute variables)
    ↓
Lower (normalize, resolve refs)
    ↓
Validate (schema, dimensions)
    ↓
Build (construct PyTorch modules)
    ↓
Run (execute training)
    ↓
Verify (compare outputs)
    ↓
Benchmark (measure perplexity, latency, memory)
    ↓
Artifacts (CSV, PNG, LaTeX)
```

### Core Components

1. **Config System** ([config/](config/)) - Typed Pydantic models + 50 preset YAML manifests
2. **Compiler** ([compiler/](compiler/)) - Manifest → executable plan pipeline
3. **Topology System** ([topology/](topology/)) - Graph node types for architecture composition
4. **Layer Library** ([layer/](layer/)) - ~35 PyTorch module types
5. **Training Modes** ([trainer/](trainer/)) - Standard, upcycle, orchestrated training
6. **Inference Engine** ([infer/](infer/)) - Generation, speculative decoding, KV-cache management
7. **Optimization Kernels** ([optimizer/](optimizer/)) - Triton (CUDA) and Metal (Apple Silicon)
8. **AI Agent System** ([ai/](ai/)) - Multi-agent collaboration for research workflows

---

## Key Technologies

### ML Stack

- **PyTorch 2.9.1** - Deep learning framework
- **Transformers 4.57.3** - HuggingFace model support
- **NumPy 2.4.0** - Numerical computing
- **TensorBoard & Weights & Biases** - Experiment tracking

### Optimization & Performance

- **Triton** - GPU kernel programming (CUDA-optimized attention, normalization, SSM kernels)
- **Metal** - Apple Silicon GPU kernels (MPS/Metal optimization for M-series chips)
- **torch.compile** - Graph optimization and compilation
- **FalkorDB** - GraphBLAS-based graph database for AI reasoning

### AI & Agents

- **Google Agent Development Kit (ADK)** - Multi-agent framework
- **LiteLLM** - LLM provider abstraction (OpenAI, Anthropic Claude, Google Gemini)
- **Model Context Protocol (MCP)** - Tool/agent integration standard
- **Graphiti** - Temporal knowledge graphs with semantic memory
- **DeepLake** - Vector store for code/research data RAG

### Web & API

- **FastAPI 0.123.10** - REST API server
- **Uvicorn** - ASGI server
- **Pydantic 2.12.5** - Data validation
- **Click** - CLI framework
- **Textual** - TUI framework for interactive dashboards

---

## Directory Structure

```
caramba/
├── __main__.py              # CLI entry point (routes to cli.py)
├── cli.py                   # Main CLI commands (221 lines)
├── pyproject.toml           # Package definition & dependencies
├── Makefile                 # Development commands
├── docker-compose.yml       # Service definitions
│
├── config/                  # Typed Pydantic models + preset manifests
│   ├── presets/             # 49 ready-to-run experiments (DBA, Llama, MoE, Mamba, ViT)
│   ├── benchmarks/          # Benchmark configurations
│   ├── manifest.py          # Main manifest schema (v2)
│   ├── model.py             # Model configuration schema
│   ├── layer.py             # Layer configuration schema
│   └── *.py                 # Other config schemas
│
├── compiler/                # Manifest → executable plan pipeline
│   ├── lower.py             # YAML normalization & reference resolution
│   ├── validate.py          # Schema validation & dimension checking
│   └── plan.py              # Compilation planning
│
├── topology/                # Graph node types for architecture composition
│   ├── stacked.py           # Sequential layer execution
│   ├── residual.py          # Skip connections
│   ├── nested.py            # Repeated layers (N blocks)
│   ├── parallel.py          # Multi-head patterns
│   ├── branching.py         # Feature fusion
│   ├── cyclic.py            # Graph connections
│   └── recurrent.py         # RNN-style cache management
│
├── layer/                   # ~35 PyTorch module implementations
│   ├── attention/           # Multi-head, GQA, DBA (Decoupled Bottleneck)
│   │   ├── dba.py           # Decoupled Bottleneck Attention
│   │   ├── multi_head.py    # Standard multi-head attention
│   │   └── grouped_query.py # Grouped query attention
│   ├── ssm.py               # Selective State Space (Mamba-style)
│   ├── moe.py               # Mixture of Experts with load balancing
│   ├── glu.py               # SwiGLU, GEGLU variants
│   ├── lora.py              # Low-rank adaptation
│   ├── rope.py              # Rotary position embeddings
│   ├── diffusion_head.py    # Denoising head for diffusion
│   ├── graph_conv.py        # Graph convolution
│   └── rms_norm.py, layer_norm.py  # Normalization layers
│
├── model/                   # Model assembly & utilities
│   ├── build.py             # Construct PyTorch from topology
│   ├── trace.py             # Activation tracing for verification
│   └── embedders.py         # Token/position embeddings
│
├── trainer/                 # Training modes (41 files)
│   ├── standard.py          # End-to-end training (70KB file)
│   ├── upcycle.py           # Architecture surgery + distillation
│   ├── orchestrated.py      # Dynamic optimizer switching (SWATS, PIDAO)
│   ├── objectives.py        # Loss functions (cross-entropy, diffusion, etc.)
│   ├── distributed.py       # DDP/FSDP multi-GPU
│   ├── blockwise.py         # Blockwise gradients
│   └── steppers/            # Optimizer implementations
│
├── infer/                   # Generation & inference (23 files)
│   ├── generate.py          # Main generation loop (30KB)
│   ├── speculative.py       # Speculative decoding
│   ├── cache_policy.py      # KV-cache quantization & management
│   ├── cache_plan.py        # Budget-aware cache strategies
│   └── autonomous_runtime.py # Autonomous inference
│
├── cache/                   # KV-cache implementations
│   ├── decoupled.py         # Decoupled KV-cache for DBA
│   ├── tensor.py            # Quantized tensor caching
│   └── layer.py             # Layer-specific caching
│
├── optimizer/               # Kernel implementations (44 files, 730KB)
│   ├── triton/              # CUDA-optimized kernels
│   │   ├── dba_attention_triton.py      # DBA attention kernels
│   │   ├── flash_attention_triton.py    # Flash Attention
│   │   ├── rmsnorm_triton.py
│   │   ├── layernorm_triton.py
│   │   ├── adamw_triton.py              # Fused AdamW
│   │   └── rope_triton.py
│   ├── metal/               # Apple Metal kernels (34 files, .metal & .mm)
│   │   └── Various Metal implementations for MPS
│   └── kernel_registry.py   # Kernel selection & registration
│
├── benchmark/               # Evaluation & metrics (23 files)
│   ├── perplexity.py        # Language modeling evaluation
│   ├── latency.py           # Inference speed measurements
│   ├── memory.py            # Peak memory, KV-cache tracking
│   ├── behavior.py          # Model behavior benchmarks
│   ├── artifacts.py         # Paper-ready outputs (CSV, PNG, LaTeX)
│   └── runner.py            # Benchmark orchestration
│
├── experiment/              # Unified pipeline orchestration
│   ├── runner.py            # Entry point for manifest execution
│   ├── group.py             # Experiment grouping
│   └── paper_artifacts.py   # Publication artifact generation
│
├── ai/                      # AI agents & automation (12 files)
│   ├── agent.py             # Google ADK wrapper (640+ lines)
│   ├── persona.py           # Agent personality/role definitions
│   ├── a2a_server.py        # A2A protocol handling
│   ├── tools/               # MCP tool implementations
│   │   ├── deeplake/        # Vector store access
│   │   ├── colbert/         # Semantic search
│   │   ├── webcrawl/        # Web research
│   │   └── ...
│   └── process/             # Process types
│       ├── brainstorm.py     # Multi-agent discussion
│       ├── development.py    # Development workflows
│       └── manifest.py       # Manifest generation
│
├── codegraph/               # Code RAG infrastructure
│   ├── parser.py            # AST parsing for codebase
│   └── sync.py              # Sync to FalkorDB graph
│
├── runtime/                 # Execution environment (16 files)
│   ├── engine.py            # TorchEngine (component resolution)
│   ├── readiness.py         # Target validation
│   └── trace/               # Execution tracing
│
├── data/                    # Dataset loading (26 files)
│   ├── datasets/            # Dataset implementations
│   ├── tokenizers/          # Tokenizer support
│   ├── transforms/          # Data transforms
│   ├── code_chunks.py       # Code chunking for RAG
│   ├── event_trace.py       # Event-based data streaming
│   └── mosaic_synth.py      # Synthetic data generation
│
├── instrumentation/         # Logging & monitoring
│   ├── JSONL/HDF5 logging
│   ├── TensorBoard hooks
│   └── Weights & Biases integration
│
├── console/                 # Rich-based logging
├── api/                     # REST API (FastAPI)
├── tui/                     # Terminal UI (Textual)
├── docs/                    # Comprehensive markdown docs (10 files)
├── carmath/                 # Reusable mathematical utilities
├── resonant/                # Associative memory implementation
├── orchestrator/            # Dynamic optimization orchestration
└── loader/                  # Model & checkpoint loading
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- CUDA toolkit (for NVIDIA GPUs) or Metal (for Apple Silicon)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/theapemachine/caramba.git
cd caramba

# Install with uv (recommended)
make install

# Or manually with pip
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Quick Start

```bash
# Run a preset experiment
caramba run config/presets/llama32_1b_dba.yml

# Run with dry-run to see compilation output
caramba run config/presets/llama32_1b_dba.yml --dry-run

# Run specific target from manifest
caramba run my_manifest.yml --target my_experiment

# Launch TUI dashboard
caramba tui

# Start API server
caramba serve --host 0.0.0.0 --port 8000
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# LLM API Keys (for AI agents)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Experiment Tracking
WANDB_API_KEY=your_key_here

# FalkorDB
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# DeepLake
DEEPLAKE_TOKEN=your_token_here
```

---

## Core Concepts

### 1. Manifests (YAML Configuration)

Manifests are the declarative heart of Caramba. Everything is YAML-configurable.

**Structure (Version 2):**

```yaml
version: 2
name: my_experiment

targets:
  experiment_name:
    type: experiment
    system:
      # Model architecture definition
    data:
      # Dataset configuration
    objective:
      # Loss function
    trainer:
      # Training strategy
    runs:
      # Training phases
    benchmarks:
      # Evaluation configurations
```

**Example:** See [config/presets/llama32_1b_dba.yml](config/presets/llama32_1b_dba.yml:1)

### 2. Topology System

Layers compose via **Topology** nodes (graph-based architecture):

- **StackedTopology** - Sequential execution (`layer1 → layer2 → layer3`)
- **ResidualTopology** - Skip connections (`x + layer(x)`)
- **NestedTopology** - Repetition (`[layer] × N`)
- **ParallelTopology** - Parallel execution (multi-head patterns)
- **BranchingTopology** - Concatenation/fusion
- **CyclicTopology** - Cyclic connections
- **RecurrentTopology** - Cache management (RNN-style)

**Implementation:** See [topology/](topology/)

### 3. Layer Types

Caramba provides ~35 layer types:

- **Attention:** Multi-head, GQA, DBA (Decoupled Bottleneck)
- **SSM:** Selective State Space (Mamba-style)
- **MoE:** Mixture of Experts with load balancing
- **Activations:** SwiGLU, GEGLU, ReLU, GELU
- **Normalization:** RMSNorm, LayerNorm
- **Embeddings:** Token, Position, RoPE
- **Other:** LoRA, Diffusion heads, Graph convolution

**Implementation:** See [layer/](layer/)

### 4. Training Modes

- **Standard** ([trainer/standard.py](trainer/standard.py:1)) - Scratch training with optimization
- **Upcycle** ([trainer/upcycle.py](trainer/upcycle.py:1)) - Convert pretrained models via architecture surgery
- **Orchestrated** ([trainer/orchestrated.py](trainer/orchestrated.py:1)) - Dynamic optimizer switching

### 5. KV-Cache Management

Budget-aware quantization strategies:

- **Precision:** fp32, fp16, q8, q4
- **Policies:** Full, quantized, decoupled (for DBA)
- **Planning:** Automatic budget calculation based on memory constraints

**Implementation:** See [cache/](cache/) and [infer/cache_policy.py](infer/cache_policy.py:1)

### 6. Platform-Specific Optimization

- **Apple Silicon (MPS)** - Native Metal kernels ([optimizer/metal/](optimizer/metal/))
- **NVIDIA (CUDA)** - Triton optimization ([optimizer/triton/](optimizer/triton/))
- **CPU** - Full functionality fallback

---

## Development Guidelines

### Code Style

- **Type hints:** Use throughout (Pydantic models for configs)
- **Docstrings:** Google-style docstrings
- **Formatting:** Black, isort
- **Linting:** Ruff, mypy

### Adding New Layers

1. Create new layer class in [layer/](layer/)
2. Inherit from `torch.nn.Module`
3. Add configuration schema to [config/layer.py](config/layer.py:1)
4. Register in layer factory
5. Add tests
6. Update documentation

**Example:**

```python
# layer/my_layer.py
import torch
import torch.nn as nn

class MyLayer(nn.Module):
    """
    My custom layer implementation.

    Args:
        dim: Input/output dimension
        config: Layer configuration
    """
    def __init__(self, dim: int, config: dict):
        super().__init__()
        self.dim = dim
        # Initialize parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation
        return x
```

### Adding New Topologies

1. Create new topology class in [topology/](topology/)
2. Inherit from `Topology` base class
3. Implement `forward()` method
4. Add configuration schema
5. Register in topology factory
6. Add tests

### Adding Optimization Kernels

**Triton (CUDA):**

1. Create kernel in [optimizer/triton/](optimizer/triton/)
2. Use `@triton.jit` decorator
3. Add wrapper function
4. Register in [optimizer/kernel_registry.py](optimizer/kernel_registry.py:1)
5. Add benchmarks

**Metal (Apple Silicon):**

1. Create `.metal` shader in [optimizer/metal/](optimizer/metal/)
2. Create Objective-C++ wrapper (`.mm` file)
3. Add Python bindings
4. Register in kernel registry
5. Add benchmarks

### Testing Best Practices

- **Unit tests:** Test individual components
- **Integration tests:** Test pipelines
- **Property tests:** Test invariants
- **Benchmark tests:** Track performance regressions

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test file
pytest tests/test_my_feature.py

# Run with markers
pytest -m "not slow"
```

---

## AI Agent System

### Overview

Caramba includes a sophisticated AI agent system built on:

- **Google Agent Development Kit (ADK)** - Multi-agent framework
- **Model Context Protocol (MCP)** - Tool integration
- **Graphiti** - Temporal knowledge graphs
- **DeepLake** - Code/research RAG
- **FalkorDB** - Graph database for reasoning

**Implementation:** See [ai/](ai/)

### Agent Architecture

```python
# ai/agent.py
class Agent:
    """Wrapper around Google Agent Development Kit"""

    def __init__(self, name: str, persona: str, model: str = "claude-sonnet-3-5"):
        # Initialize agent with persona and tools

    async def run(self, task: str) -> str:
        # Execute task with tool access
```

### Process Types

- **Brainstorm** ([ai/process/brainstorm.py](ai/process/brainstorm.py:1)) - Multi-agent architecture discussion
- **Development** ([ai/process/development.py](ai/process/development.py:1)) - Experiment refinement
- **Manifest** ([ai/process/manifest.py](ai/process/manifest.py:1)) - Auto-generation of experiments

### Available Tools (MCP)

- **DeepLake** ([ai/tools/deeplake/](ai/tools/deeplake/)) - Vector store access for code/research
- **ColBERT** ([ai/tools/colbert/](ai/tools/colbert/)) - Semantic search
- **WebCrawl** ([ai/tools/webcrawl/](ai/tools/webcrawl/)) - Web research & paper parsing
- **FalkorDB** - Graph database queries

### Running Agent Processes

```bash
# Multi-agent brainstorm
make brainstorm

# With FalkorDB graph memory
make brainstorm-full

# Index codebase to FalkorDB
make ingest
caramba codegraph-sync /path/to/repo
```

---

## Testing & Quality

### Test Structure

```
tests/
├── unit/              # Component tests
├── integration/       # Pipeline tests
├── benchmarks/        # Performance tests
└── fixtures/          # Test data
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=caramba --cov-report=html

# Specific module
pytest tests/unit/test_compiler.py

# With markers
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"       # Only GPU tests
```

### Quality Tools

```bash
# Type checking
mypy caramba/

# Linting
ruff check caramba/

# Formatting
black caramba/
isort caramba/
```

---

## Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Services:
# - webcrawl-mcp (port 3000)
# - falkordb (port 6379)
# - graphiti-mcp (port 3001)
# - sympy-mcp (port 3002)

# Stop services
docker-compose down
```

### API Server

```bash
# Start FastAPI server
caramba serve --host 0.0.0.0 --port 8000

# Access API docs
open http://localhost:8000/docs
```

### TUI Dashboard

```bash
# Launch terminal UI
caramba tui --url ws://localhost:8000/ws --log experiments.log
```

---

## Common Tasks

### Running Experiments

```bash
# Run preset
caramba run config/presets/llama32_1b_dba.yml

# Run with specific target
caramba run my_manifest.yml --target experiment_1

# Dry run (see compilation output)
caramba run my_manifest.yml --dry-run
```

### Benchmarking

```bash
# Full benchmark suite
make paper  # Runs DBA upcycle pipeline with benchmarks

# Individual benchmarks are defined in manifests:
benchmarks:
  - type: perplexity
    dataset: wikitext103_valid
  - type: latency
    prompt_lengths: [128, 256, 512, 1024]
  - type: memory
    sequence_lengths: [512, 1024, 2048]
```

### Code Indexing

```bash
# Index codebase to FalkorDB (for AI agents)
make ingest

# Or manually
caramba codegraph-sync /path/to/repo --db caramba_graph
```

### Creating Custom Experiments

1. Start with a preset as template
2. Modify architecture in `system:` section
3. Adjust training parameters in `trainer:` section
4. Add benchmarks in `benchmarks:` section
5. Run with `caramba run my_manifest.yml`

**Example manifest:**

```yaml
version: 2
name: my_custom_experiment

targets:
  main:
    type: experiment

    system:
      layers:
        - type: embedding
          name: embed
          vocab_size: 50000
          dim: 512

        - type: rms_norm
          name: norm_in
          dim: 512

        - type: attention
          subtype: multi_head
          name: attn
          dim: 512
          heads: 8

        - type: rms_norm
          name: norm_out
          dim: 512

        - type: linear
          name: head
          in_dim: 512
          out_dim: 50000

      topology:
        type: stacked
        children:
          - embed
          - norm_in
          - attn
          - norm_out
          - head

    data:
      type: wikitext103
      batch_size: 32
      sequence_length: 512

    objective:
      type: cross_entropy

    trainer:
      type: standard
      optimizer: adamw
      learning_rate: 1e-4
      max_steps: 10000

    runs:
      - name: main_run
        steps: 10000

    benchmarks:
      - type: perplexity
        dataset: wikitext103_valid
      - type: latency
        prompt_lengths: [128, 512]
```

### Viewing Results

Experiment artifacts are saved to:

```
experiments/
└── <experiment_name>/
    ├── checkpoints/
    ├── logs/
    ├── artifacts/
    │   ├── results.csv
    │   ├── perplexity.png
    │   ├── latency.png
    │   ├── memory.png
    │   └── results.tex
    └── config.yml
```

---

## Additional Resources

### Documentation

- [Architecture Guide](docs/architecture.md)
- [Manifest v2 Guide](docs/manifest_v2.md)
- [DBA Attention](docs/dba.md)
- [Training Modes](docs/training.md)
- [Benchmarking](docs/benchmarking.md)

### Key Files to Understand

1. [cli.py](cli.py:1) - CLI commands and entry points
2. [experiment/runner.py](experiment/runner.py:1) - Main execution pipeline
3. [compiler/lower.py](compiler/lower.py:1) - Manifest normalization
4. [model/build.py](model/build.py:1) - PyTorch model construction
5. [trainer/standard.py](trainer/standard.py:1) - Training implementation
6. [ai/agent.py](ai/agent.py:1) - AI agent system

### Contact & Support

- **Repository:** https://github.com/theapemachine/caramba
- **Issues:** https://github.com/theapemachine/caramba/issues
- **Documentation:** [docs/](docs/)

---

## Changelog

### Version 2.0 (Current)

- Declarative manifest system (v2)
- Multi-agent AI collaboration
- FalkorDB + Graphiti integration
- Metal kernel optimization for Apple Silicon
- Decoupled Bottleneck Attention (DBA)
- Orchestrated training with dynamic optimization
- Publication-ready artifact generation

### Previous Versions

- v1.x - Initial implementation with basic training loops
- v0.x - Prototype research codebase

---

**Note:** This guide is intended for AI assistants (like Claude) working with the Caramba codebase. It provides context on architecture, conventions, and common patterns to facilitate effective code assistance.
