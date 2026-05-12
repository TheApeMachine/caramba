# Manifest & Governance

---

## Two Sources of Truth

Caramba's architecture rests on a clean split between two immutable anchors:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MANIFEST (driver)                                                         │
│   The upstream source of truth: what should happen.                        │
│                                                                             │
│   • Immutable contract for a research project and its runs                  │
│   • All decisions derived from the manifest, never from ad-hoc flags        │
│   • No silent fallbacks: if it isn't in the manifest, it doesn't exist     │
│   • Declares: architecture, datasets, training schedule, targets,           │
│     resource constraints, evaluation criteria                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ executes
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MODEL (collector)                                                         │
│   The downstream source of truth: what did happen.                         │
│                                                                             │
│   • Stateful sink for outcomes produced by executing the plan               │
│   • Step counters, loss, metrics, checkpoints, artifacts, traces            │
│   • Append-only timeline of events                                          │
│   • Pointers to immutable artifacts (content-addressed)                     │
│   • Immutable once committed. Self-describing. Fully auditable.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Between them lives the Experiment—a transient workspace that accumulates results until the Notary verifies them and commits a new Model version.

---

## Contract Chain

```
MANIFEST (driver) → PROTOCOL (actor) → MODEL (collector)
```

| Contract     | Role                             | Contains                                                        |
|--------------|----------------------------------|-----------------------------------------------------------------|
| **Manifest** | Declares the researcher's intent | What we're trying to learn                                      |
| **Protocol** | Defines procedures for execution | How to execute; embeds the Manifest                             |
| **Model**    | Outcome *and* ledger             | Architecture, weights, full audit trail, embeds everything      |

Each level accumulates the one before it. The Model isn't just the output—it is the complete provenance chain.

---

## Atomic Intent

A Manifest declares a unit of scientific intent. The comparison is atomic—all variants and seeds belong to one intent. If any part fails, the whole intent is voided and must be resubmitted.

```yaml
name: dba_ablation_study
description: Compare DBA variants against baseline across multiple seeds

variants:
  - baseline
  - bottleneck
  - decoupled
  - gqa

seeds:
  - 1337
  - 1338
  - 1339

system:
  topology:
    type: GraphTopology
    inputs: [x]
    nodes:
      - id: embed
        op: embedding.token
        in: [x]
        out: [h]
        config:
          vocab_size: 50257
          d_model: 768
      # ... rest of topology

trainer:
  optimizer: adam
  learning_rate: 3.0e-4
  max_steps: 10000
  warmup_steps: 1000

datasets:
  train:
    source: s3://my-bucket/train
    format: arrow
```

**Why all-or-nothing?**

Consider the alternative: you run 12 experiments, 3 fail due to a config bug, you comment them out, re-run just those 3. Now you have results from two sessions, possibly different code versions. Six months later: *"wait, were these all from the same sweep?"*

A quick void and clean re-run is nearly always cheaper than discovering later that your published comparison had a confound.

---

## Manifest Compilation

**Package:** `pkg/manifest`

The compiler is a chain of stages that transforms YAML into an executable IR graph. The pipeline is strict—it fails immediately on any invalid structure.

### Stages

```
YAML file
    ↓
pkg/manifest/parser.go          — parse YAML into typed structs
    ↓
pkg/manifest/compiler.go        — variable substitution, ${var} expansion
                                  repeat unrolling (repeat: N blocks)
                                  include resolution
    ↓
pkg/manifest/registry.go        — resolve operation identifiers
                                  validate all ops exist
    ↓
pkg/manifest/graph.go           — build GraphTopology DAG
                                  check for cycles, missing inputs
    ↓
pkg/backend/compute/ir/         — lower to hardware-agnostic IR
    ↓
pkg/backend/compute/orchestrator/ — CSE, DCE, fusion, schedule
    ↓
pkg/backend/compute/runner/     — dispatch to hardware
```

### Variable Substitution

Variables declared at the top of a manifest are substituted throughout:

```yaml
variables:
  d_model: 768
  n_heads: 12
  vocab_size: 50257

system:
  topology:
    nodes:
      - id: attn
        op: attention.sdpa
        config:
          d_model: ${d_model}
          n_heads: ${n_heads}
```

### Repeat Unrolling

Repeated blocks (e.g., transformer layers) are declared once and expanded:

```yaml
nodes:
  - repeat: 12
    index: layer_idx
    template:
      - id: layer_${layer_idx}_attn
        op: attention.sdpa
        in: [layer_${layer_idx}_input]
        out: [layer_${layer_idx}_attn_out]
      - id: layer_${layer_idx}_ffn
        op: projection.linear
        in: [layer_${layer_idx}_attn_out]
        out: [layer_${next_layer_idx}_input]
```

### Graph Topology

The final compiled form is a `GraphTopology`—a flat DAG of operations with explicit input/output bindings:

```go
// pkg/manifest/graph.go

type GraphTopology struct {
    Inputs  []string
    Nodes   []Node
    Outputs []string
}

type Node struct {
    ID     string
    Op     string
    In     []string
    Out    []string
    Config map[string]any
}
```

---

## Experiment Lifecycle

```
Model(v0) ──┐
            │
            ▼
     ┌─────────────┐
     │ Experiment  │◀── Manifest (intent)
     │             │◀── Protocol (rules)
     │  (working)  │
     └──────┬──────┘
            │
            │  execute()
            ▼
     ┌─────────────┐
     │   Notary    │
     │  validate   │
     └──────┬──────┘
            │
      ┌─────┴─────┐
      ▼           ▼
   VERIFIED     VOIDED
      │           │
      ▼           ▼
   commit()    discard
      │           │
      ▼           ▼
Model(v1)    Model(v0) unchanged
```

The Experiment starts as a copy of the current Model state. It accumulates updates—weights, metrics, artifacts, ledger entries—during execution. If verified, it becomes the new Model. If voided, it is discarded and the original Model is untouched.

---

## Operation Registry

Operations are registered by identifier string. The manifest compiler resolves identifiers to operation implementations at compile time:

```go
// pkg/manifest/registry.go

type Registry struct {
    ops map[string]OperationSpec
}

func (registry *Registry) Register(id string, spec OperationSpec) {
    registry.ops[id] = spec
}

func (registry *Registry) Resolve(id string) (OperationSpec, error) {
    spec, ok := registry.ops[id]
    if !ok {
        return OperationSpec{}, fmt.Errorf("unknown operation: %q", id)
    }
    return spec, nil
}
```

All operations in `pkg/asset/template/operation/` have corresponding registry entries. Adding a new operation requires both an implementation in `pkg/backend/compute/cpu/operation/` and a YAML schema in `pkg/asset/template/operation/`.

---

## YAML Schema Reference

A complete manifest has the following top-level structure:

```yaml
# Identity
name: string                      # required
description: string               # optional
version: string                   # optional

# Variables (substituted throughout)
variables:
  key: value

# Dataset sources
datasets:
  train:
    source: string                # s3://, local path, etc.
    format: string                # arrow, jsonl, parquet, ...
  eval:
    source: string
    format: string

# Training configuration
trainer:
  optimizer: string               # adam, sgd, lion, ...
  learning_rate: float
  max_steps: int
  warmup_steps: int
  batch_size: int
  gradient_clip: float

# Model architecture
system:
  topology:
    type: GraphTopology
    inputs: [string]
    nodes:
      - id: string
        op: string                # e.g. attention.sdpa
        in: [string]
        out: [string]
        config:
          key: value
    outputs: [string]

# Evaluation targets
targets:
  - name: string
    dataset: string
    metric: string

# Secrets (values never stored in manifest)
secrets:
  - name: string
    env: string                   # environment variable name
```

---

## Design Rules

- **Manifest is upstream truth.** No environment-variable config, no CLI flags for core behavior.
- **Model is downstream truth.** Results come from execution; never recompute history from logs.
- **Fail hard and fast.** If plan/record invariants are violated, fail immediately with a clear error.
- **No silent fallbacks.** If a field is required and missing, the compiler errors out.
