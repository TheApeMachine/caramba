# Architecture

Caramba is organized as a distributed actor system. Each major concern is an independent entity that communicates via typed messages and can run anywhere—same process, different machine, or cloud cluster.

---

## Three Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CLUSTER (infrastructure)                                                  │
│   "What machines exist and how do they connect?"                            │
│                                                                             │
│   Node discovery, registration, health monitoring, network topology.        │
│   This is the plumbing. It doesn't know about experiments or models.        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ provides connectivity
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ACTORS (logical entities)                                                 │
│   "What work needs to be done?"                                             │
│                                                                             │
│   Notary, Experiment, Model, Storage, Backend.                              │
│   Can be instantiated on any node. Don't care about physical location.      │
│   Communicate via typed messages.                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ needs placement decisions
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ORCHESTRATOR (scheduler/router)                                           │
│   "Where should work happen?"                                               │
│                                                                             │
│   Resource matching, job queuing, load balancing, lease management.         │
│   Knows cluster state and job requirements; makes placement decisions.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
caramba/
├── cmd/                          # CLI entrypoints
│   ├── root.go                   # Root command
│   ├── serve.go                  # caramba serve
│   ├── research.go               # caramba research run|status|inspect
│   └── asset/                    # caramba asset list|show
│
├── pkg/
│   ├── manifest/                 # Manifest compilation pipeline
│   │   ├── parser.go             # YAML → typed structs
│   │   ├── compiler.go           # Variable substitution, topology lowering
│   │   ├── registry.go           # Operation registry
│   │   ├── graph.go              # Topology graph
│   │   ├── ops.go                # Operation descriptors
│   │   └── experiment.go         # Experiment lifecycle
│   │
│   ├── backend/
│   │   ├── api/                  # HTTP API server (Fiber)
│   │   ├── architecture/         # Architecture management
│   │   ├── modelscope/           # Model registry / hub
│   │   └── compute/              # Compute backends
│   │       ├── ir/               # Hardware-agnostic IR (Graph + Node)
│   │       ├── tensor/           # Tensor abstraction + kernel interface
│   │       ├── orchestrator/     # IR optimization passes
│   │       │   ├── cse.go        # Common subexpression elimination
│   │       │   ├── dce.go        # Dead code elimination
│   │       │   ├── fusion.go     # Operator fusion
│   │       │   └── scheduler.go  # Topological scheduling
│   │       ├── runner/           # Runner interface
│   │       ├── cpu/              # Go + SIMD/Assembly backend
│   │       │   ├── operation/    # Per-op CPU implementations
│   │       │   ├── optimizer/    # Optimizer kernels (Adam, SGD, ...)
│   │       │   └── block/        # Block schema service
│   │       ├── cuda/             # CUDA backend (Linux + CGO)
│   │       ├── metal/            # Metal backend (macOS + CGO)
│   │       └── xla/              # XLA/PJRT backend
│   │
│   ├── asset/                    # Embedded YAML templates
│   │   └── template/
│   │       ├── block/            # Pre-wired operation blocks
│   │       ├── model/            # Full model templates (LLM, vision, audio, ...)
│   │       ├── operation/        # Per-operation YAML schemas
│   │       ├── optimizer/        # Optimizer YAML schemas
│   │       └── manifest/         # Manifest templates
│   │
│   ├── store/                    # Storage adapters
│   │   ├── s3/
│   │   ├── elasticsearch/
│   │   ├── neo4j/
│   │   ├── qdrant/
│   │   └── deeplake/
│   │
│   └── config/                   # Configuration loading (Viper)
│
└── frontend/                     # Browser UI (TanStack Start + Router)
```

---

## Execution Flow

When a manifest is submitted:

```
┌──────────────┐
│ caramba run  │
│ manifest.yml │
└──────┬───────┘
       │
       ▼
┌──────────────┐     validate manifest      ┌──────────────┐
│  EXPERIMENT  │◀──────────────────────────▶│    NOTARY    │
│   (actor)    │                            │   (actor)    │
└──────┬───────┘                            └──────────────┘
       │
       │  compile manifest → IR graph
       ▼
┌──────────────┐
│  COMPILER    │  parse → lower → validate → build
│              │
│  YAML ──▶ GraphTopology ──▶ IR Graph
└──────┬───────┘
       │
       │  optimize IR
       ▼
┌──────────────┐
│ ORCHESTRATOR │  CSE → DCE → Fusion → Schedule
└──────┬───────┘
       │
       │  dispatch to runner
       ▼
┌──────────────┐     "where should these run?"     ┌──────────────┐
│   SCHEDULER  │◀──────────────────────────────────▶│   CLUSTER    │
└──────┬───────┘                                    └──────────────┘
       │
       ▼
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │  CPU Runner  │   │ CUDA Runner  │   │ Metal Runner │   ...     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         │                  │                   │                   │
│         └──────────────────┴───────────────────┘                   │
│                             │                                      │
│                        ┌────┴─────┐                                │
│                        │ STORAGE  │                                │
│                        │ (S3, ...) │                                │
│                        └──────────┘                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Compiler Pipeline

The manifest compiler is a chain of stages that transforms YAML into an executable IR graph:

```
YAML Manifest
    ↓
Parse (pkg/manifest/parser.go)
    ↓
Variable Substitution / Repeat Expansion (pkg/manifest/compiler.go)
    ↓
Operation Registry Resolution (pkg/manifest/registry.go)
    ↓
GraphTopology (pkg/manifest/graph.go)
    ↓
IR Graph (pkg/backend/compute/ir/)
    ↓
Optimizer Passes (pkg/backend/compute/orchestrator/)
    │  ├── CSE   (common subexpression elimination)
    │  ├── DCE   (dead code elimination)
    │  ├── Fusion (operator fusion)
    │  └── Schedule (topological ordering)
    ↓
Runner Dispatch (pkg/backend/compute/runner/)
```

The compiler is intentionally strict—it fails hard on any type mismatch or invalid structure. Problems are caught at compile time, not during training.

---

## IR Graph

The intermediate representation (`pkg/backend/compute/ir`) is a DAG of typed nodes. It is hardware-agnostic: the same graph structure is passed to CPU, CUDA, Metal, and XLA runners.

```go
// Every computation is a Node in the graph
type Node struct {
    ID       string
    Op       string
    Inputs   []*Node
    Config   map[string]any
}

// The graph is the complete computation
type Graph struct {
    Nodes   []*Node
    Inputs  []*Node
    Outputs []*Node
}
```

---

## Tensor Ownership

Tensors are always owned by a specific backend. They are never implicitly copied between backends. The only way to move data is through an explicit download:

```go
type Float64Tensor interface {
    Shape() []int
    DownloadFloat64() ([]float64, error)
    Location() Location   // "cpu", "cuda:0", "metal", "xla:cpu", etc.
}
```

This design eliminates a major class of bugs where implicit host-device copies silently tank performance or produce incorrect results.

---

## Orchestrator Passes

Before dispatch, the IR graph is optimized by four passes:

| Pass    | What it does                                                                      |
|---------|-----------------------------------------------------------------------------------|
| **CSE** | Identifies nodes with identical ops and inputs; replaces duplicates with one node |
| **DCE** | Removes nodes whose outputs are never consumed                                    |
| **Fusion** | Merges compatible adjacent ops into a single fused kernel (e.g., matmul+bias+gelu) |
| **Schedule** | Produces a valid topological execution order                                 |

---

## Distributed Model

Each actor is location-transparent. In a single-machine setup, all actors run in the same process. In a distributed setup, they communicate over the network with no change to the actor logic.

The key invariant: **actors emit evidence, they don't assert global truth.** Only the Notary answers questions about what is currently valid.

→ [The Notary](./notary.md)
