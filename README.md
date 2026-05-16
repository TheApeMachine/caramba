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

# Start a terminal chat shell
caramba chat

# Submit a manifest
caramba research run manifest.yaml

# Inspect assets
caramba asset list
```

→ [Getting Started Guide](./docs/getting-started.md)

---

## Hub Assets

Caramba resolves Hugging Face Hub assets through a revision-aware local cache
that mirrors the smooth `hf_hub_download` and `snapshot_download` workflow while
keeping provenance explicit. Hub settings live under `hub` in
`cmd/asset/config.yml`, including endpoint, cache directory, optional token,
offline mode, worker count, and Xet enablement.

Model loaders can use plain repo IDs like `openai-community/gpt2` or explicit
locators like `hf://model/openai-community/gpt2@main`. The cache records refs,
commit-pinned snapshots, content-addressed blobs, and metadata for every pulled
file. Xet-backed files are detected from Hub resolve headers and reconstructed
through CAS when `hub.xet.active` is enabled.

Tokenizer operations use the same Hub cache. `tokenizer.load` registers a
`tokenizer.json` artifact, `tokenizer.encode` turns prompt text into token IDs
that can feed `embedding.token`, and `tokenizer.decode` turns generated IDs back
into text. The first tokenizer backend is Hugging Face ByteLevel BPE, covering
GPT-style decoder models without delegating tokenization to Python.

The `chat` command provides the terminal prompt and streaming path for local
model interaction:

```bash
caramba chat
caramba chat --manifest model/llm/llama-3-2-1b-instruct.yml
```

`chat` accepts only a manifest selector. Runtime, compute backend, model source,
tokenizer source, Hub repo type/revision/cache overrides, and generation policy
live in the manifest under `system.runtime`; architecture still lives under
`system.topology`. The default manifest is `model/llm/gpt2.yml`, and Llama 3.2
1B Instruct is declared in `model/llm/llama-3-2-1b-instruct.yml`.

When started, `chat` compiles the YAML model manifest through
`pkg/manifest.Compiler`, lowers it through `pkg/manifest` into the compute IR,
and executes through `pkg/backend/compute.Backend`. Hub files and serialized
tensor formats bind into matching manifest nodes by structure. The first weight
binder reads SafeTensors checkpoints, including sharded
`model.safetensors.index.json` layouts, and attaches `weight`/`bias` state to
the lowered IR before backend execution. The model runtime uses causal
attention, projects decode-time logits from the final token position, and
streams token selection through manifest-configured repetition penalty,
temperature, top-k, top-p, seed, stop tokens, and special-token stopping
controls. Causal attention nodes receive a per-generation KV cache: the prompt
is prefetched once, then each decode step executes a single-token graph with
absolute position IDs and cached keys/values. Metal keeps that cache in resident
GPU buffers, preallocates the generation token budget, and writes decode chunks
in place without copying K/V through host memory.

---

## Compute Backends

The compute layer is organized around explicit tensor ownership and a typed, hardware-agnostic IR. Backend kernels upload values once into a resident tensor store and only download at real boundaries. The executor releases owned dependencies after their final graph consumer, and the host arena reuses released spans instead of silently falling back to heap allocation. The IR graph now travels through a compiler pipeline with verification, canonicalization, semantic CSE, algebraic simplification, legality-aware fusion, side-effect-aware DCE, memory planning, cost scheduling, and backend lowering before dispatch.

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

The optimizer does not silently invent fused nodes. Backend capability contracts declare supported operations, legal fusion patterns, and shape-aware costs. Required operation IDs flow through an explicit dispatch contract: each backend either routes to its native `state.Operation` registry or to a resident tensor kernel for that operation, with no wildcard capability declaration.

CPU SIMD symbols are real architecture implementations, not scalar branch
aliases. Activation kernels currently include AVX2, SSE2, and arm64 NEON
assembly paths with scalar tails named by architecture rather than by a vector
ISA they do not use, and the CPU test suite rejects SIMD files that only branch
to scalar symbols.

Optimizer execution follows the same rule. CPU optimizers own AVX2, SSE2, and
NEON assembly paths; Metal and CUDA own native kernels; XLA optimizers compile
their update math to StableHLO and execute through the configured PJRT plugin.
Backend parity tests compare optimizer `state.Dict` mutations against the CPU
contract instead of accepting backend-specific drift. Reduction-heavy optimizer
steps such as LARS, LAMB, and L-BFGS execute as native reduction/apply stages on
Metal, CUDA, and XLA.
XLA VSA bundle/similarity and masking apply operations compile as complete
StableHLO modules with PJRT-managed inputs, so intermediate tensors stay inside
the selected XLA runtime.
Metal parity now covers the same contract sizes for repaired causal and
probabilistic kernels. Transposed convolution initializes bias on-device before
scatter-add, Active Inference expected free energy clamps probabilities in the
Metal kernel, DAG Markov factorization computes residual variance in Metal, and
frontdoor adjustment performs equal-frequency sorting through Metal kernels
instead of CPU-side preprocessing.

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

# XLA via PJRT (configure compute.xla in cmd/asset/config.yml first)
go test -tags "cgo xla" ./pkg/backend/compute/xla/...
```

Metal uses `//go:build darwin && cgo`—there is no separate `metal` tag; Darwin + CGO selects the Metal implementation automatically.

For XLA, PJRT include/plugin paths are loaded from `compute.xla` in `cmd/asset/config.yml`; the Go runtime resolves and passes the exact plugin file into the PJRT C layer, and direct environment-variable shadow config is intentionally not used.

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
