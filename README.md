# caramba

**A substrate for AI research**

---

## The Problem

In a typical ML workflow, state is scattered: configs live in YAML files, metrics in W&B, checkpoints on S3, decisions in Slack threads. Each handoff between systems is a point where certainty is lost.

Consider a bank counter who counts money in the vault. They know the count exactly—until they hand the money to a security guard. From that moment, they can no longer vouch for the count with absolute certainty. They didn't oversee the full chain of custody.

This is the state of ML research today. You run an experiment, and somewhere between your config file, your training script, your checkpoint, and your results table, you lose the ability to say with certainty: *"I know exactly how this model was produced."*

Caramba solves this by never letting go of the money.

---

## The Insight

**Everything accumulates forward.**

Instead of scattering state across systems, Caramba chains contracts where each accumulates into the next:

```
MANIFEST (driver) → PROTOCOL (actor) → MODEL (collector)
```

| Contract     | Role                             | Contains                                                   |
|--------------|----------------------------------|------------------------------------------------------------|
| **Manifest** | Declares the researcher's intent | What we're trying to learn                                 |
| **Protocol** | Defines procedures for execution | How to execute, embeds the Manifest                        |
| **Model**    | Outcome *and* ledger             | Architecture, weights, full audit trail, embeds everything |

The Model isn't just the output—it's the complete provenance. When you share a Model, you share its entire history. When you resume training, you resume from complete state, not a partial snapshot.

→ [Deep dive: Manifest & Governance](./docs/manifest.md)

---

## The Notary

The Notary maintains continuous custody over the entire research process. It is the single source of truth—the one system that never loses sight of the money.

```
┌────────────────────────────────────────────────────────────────┐
│                           NOTARY                               │
│                   (single source of truth)                     │
├────────────────────────────────────────────────────────────────┤
│             MANIFEST ──→ PROTOCOL ──→ MODEL                    │
│             (driver)     (actor)   (collector)                 │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                   RESEARCH PROJECT                             │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌─────────────────────────┐    ┌───────────┐  │
│  │ PROTOCOL │───▶│ EXPERIMENT              │───▶│   MODEL   │  │
│  └──────────┘    │ deploy → train → bench  │    └───────────┘  │
│                  └─────────────────────────┘          │        │
│                             │                         │        │
│                             ▼                         ▼        │
│                       ┌──────────┐             ┌──────────┐    │
│                       │ APPROVED │────────────▶│ VERIFIED │    │
│                       └──────────┘             └──────────┘    │
│                                                      │         │
│                                                      ▼         │
│                             ┌─────────────────────────────┐    │
│                             │        VALIDATION           │    │
│                             │  pass: commit to new truth  │    │
│                             │  fail: void, rollback       │    │
│                             └─────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Submit** — Researcher submits a Manifest declaring intent
2. **Validate** — Notary checks compatibility with Protocols and current Model state
3. **Execute** — Experiment runs on a copy of the Model; weights are updated in the copy only
4. **Checkpoint** — At defined points, Notary validates against Protocol expectations
5. **Commit or Void** — Pass: the copy becomes the new source of truth. Fail: the copy is destroyed, original remains untouched

→ [Deep dive: The Notary](./docs/notary.md)

---

## Atomic Intent

A Manifest declares a unit of scientific intent. Not a batch of runs—an *intent*.

```yaml
name: dba_ablation_study
variants: [baseline, bottleneck, decoupled, gqa]
seeds: [1337, 1338, 1339]
```

If you declare "compare architectures A, B, C, D across seeds 1, 2, 3," that comparison is atomic. If `bottleneck_s1338` fails, you don't have eleven good runs and one bad one. You have zero complete ablation studies.

**There is no "approved" with an asterisk.**

→ [Deep dive: Manifest & Governance](./docs/manifest.md)

---

## Quick Start

```bash
# Install
go install github.com/theapemachine/caramba@latest

# Run the API server
caramba serve

# Submit a manifest
caramba research run manifest.yaml

# Inspect assets
caramba asset list
```

→ [Getting Started Guide](./docs/getting-started.md)

---

## Compute Backends

The compute layer is organized around explicit tensor ownership and a typed, hardware-agnostic IR. Backend kernels upload values once into a resident tensor store and only download at real boundaries. The IR graph now travels through a compiler pipeline with verification, canonicalization, semantic CSE, algebraic simplification, legality-aware fusion, side-effect-aware DCE, memory planning, cost scheduling, and backend lowering before dispatch.

```
┌────────────────────────────────────────────────────────────┐
│                      COMPUTE PIPELINE                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   Manifest → Typed IR → Compiler Pipeline → Runner         │
│                                                            │
│   Runners:                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │  CPU     │  │  CUDA    │  │  Metal   │  │  XLA     │   │
│   │ (Go+SIMD)│  │ (Linux)  │  │ (macOS)  │  │ (PJRT)   │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Every backend implements the same `Runner` interface:

```go
type Runner interface {
    Execute(
      ctx context.Context, 
      graph *ir.Graph, 
      targets []*ir.Node,
    ) (map[string]tensor.Float64Tensor, error)
    
    Location() tensor.Location
    Close() error
}
```

The optimizer does not silently invent fused nodes. Backend capability contracts declare supported operations, legal fusion patterns, and shape-aware costs. CPU declares broad host operation support; CUDA and XLA only declare implemented tensor kernels; Metal declares its native operation families explicitly.

### Building Backends

```bash
# Pure Go CPU (always available)
go build ./pkg/backend/compute/cpu/...

# CPU with SIMD (AVX2/SSE2/NEON via assembly)
go build ./pkg/backend/compute/cpu/...

# CUDA (Linux, NVIDIA CUDA toolkit required)
CGO_ENABLED=1 go build -tags "cgo cuda" ./pkg/backend/compute/cuda/...
CGO_ENABLED=1 go test  -tags "cgo cuda" ./pkg/backend/compute/cuda/...

# Metal (macOS, Xcode Command Line Tools required)
CGO_ENABLED=1 go build -tags cgo ./pkg/backend/compute/metal/...
CGO_ENABLED=1 go test  -tags cgo ./pkg/backend/compute/metal/...

# XLA via PJRT (requires headers and plugin library)
export CARAMBA_XLA_INCLUDE_DIR=/path/to/xla/include
export CGO_CPPFLAGS="-I${CARAMBA_XLA_INCLUDE_DIR}"
export CARAMBA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
go test -tags "cgo xla" ./pkg/backend/compute/xla/...
```

Metal uses `//go:build darwin && cgo`—there is no separate `metal` tag; Darwin + CGO selects the Metal implementation automatically.

For XLA, `CARAMBA_PJRT_CPU_PLUGIN` and `CARAMBA_PJRT_GPU_PLUGIN` are checked first; `CARAMBA_PJRT_PLUGIN` serves as a shared fallback for single-plugin environments.

→ [Deep dive: Compute Backends](./docs/compute.md)

---

## Operations

Operations are the atomic units of computation. The library spans standard deep learning primitives through esoteric research architectures:

| Category          | Examples                                           |
|-------------------|----------------------------------------------------|
| Activation        | ReLU, GeLU, SwiGLU, Tanh, Sigmoid, Mish            |
| Attention         | SDPA, DBA, GQA, Multi-head, Causal, Flash          |
| Embedding         | Token, Positional (RoPE, ALiBi, sinusoidal)        |
| Normalization     | LayerNorm, RMSNorm, BatchNorm                      |
| Projection        | Linear, FusedQKV, LoRA                             |
| Convolution       | Conv1D, Conv2D, depthwise, grouped                 |
| Pooling           | Mean, Max, Attention pooling                       |
| Active Inference  | Free energy minimization, precision weighting      |
| Causal            | Temporal difference, causal intervention           |
| Hawkes Process    | Point process attention kernels                    |
| Markov Blanket    | Blanket detection, free energy decomposition       |
| Predictive Coding | Hierarchical prediction error                      |
| VSA               | Hyperdimensional binding, bundling, cleanup memory |

Each operation ships with CPU (Go + SIMD), CUDA, Metal, and XLA implementations. No backend falls back silently—if a kernel isn't implemented for a backend, the build fails.

→ [Deep dive: Operations](./docs/operations.md)

---

## Architecture (Node Graph Editor)

Architectures are composed visually in the browser using a node graph editor. The editor reads operation schemas from the backend, supports template blocks (pre-wired subgraphs), and serializes to YAML manifests.

```
┌──────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE EDITOR                                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Operations ──▶ Canvas (drag & connect) ──▶ YAML Manifest       │
│                        │                                         │
│                  Template Blocks                                 │
│                  (TransformerBlock, MLP, DBA, GQA, ...)         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

→ [Deep dive: Frontend & Visualization](./docs/frontend.md)

---

## Agents (Research Team)

The agent layer is the conversational ingress into the research system. It presents a multi-persona interface backed by major LLM providers (OpenAI, Anthropic, Google). Agents discuss, critique, and synthesize—but never silently modify state. Every proposal is explicit, auditable, and recorded.

→ [Deep dive: Agents](./docs/agents.md)

---

## Storage

Caramba supports pluggable storage backends:

| Backend       | Use case                              |
|---------------|---------------------------------------|
| S3            | Checkpoints, weights, large artifacts |
| Elasticsearch | Metrics, event logs, text search      |
| Neo4j         | Provenance graphs, dependency DAGs    |
| Qdrant        | Embedding similarity search           |
| DeepLake      | Tensor datasets                       |

---

## Documentation

| Document                                       | Description                                    |
|------------------------------------------------|------------------------------------------------|
| [Getting Started](./docs/getting-started.md)   | Installation, first experiment, basic workflow |
| [Architecture](./docs/architecture.md)         | System design: actors, distributed model, IR   |
| [Compute Backends](./docs/compute.md)          | CPU/SIMD, CUDA, Metal, XLA in depth            |
| [Manifest & Governance](./docs/manifest.md)    | Manifest → Protocol → Model, atomic intent     |
| [The Notary](./docs/notary.md)                 | Custody, lazy validation, the ledger           |
| [Operations](./docs/operations.md)             | Operation library, SIMD kernels, custom ops    |
| [Frontend & Visualization](./docs/frontend.md) | Node graph editor, microscope tooling          |
| [Agents](./docs/agents.md)                     | Research team interface, LLM providers         |

---

## License

MIT

---

<p align="center">
  <i>The model knows how it was made.</i>
</p>
