# Frontend & Visualization

The Caramba frontend is built on **TanStack Start + Router** and serves as the primary research interface. It covers architecture design, training visualization, and model inspection.

---

## Architecture Editor

The central tool is a node graph editor for composing model architectures visually.

```
┌──────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE EDITOR                                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┬──────────────────────────┬──────────────────┐  │
│  │              │                          │                  │  │
│  │  OPERATIONS  │       CANVAS             │   INSPECTOR      │  │
│  │              │                          │                  │  │
│  │  ▼ Attention │   ┌─────┐   ┌─────┐     │  ┌────────────┐  │  │
│  │    • SDPA    │   │  x  │──▶│ W_q │──┐  │  │ W_q        │  │  │
│  │    • GQA     │   └─────┘   └─────┘  │  │  │            │  │  │
│  │    • MQA     │       │     ┌─────┐  │  │  │ d_in: 2048 │  │  │
│  │              │       ├────▶│ W_k │──┤  │  │ d_out: 2048│  │  │
│  │  ▼ Projection│       │     └─────┘  │  │  │ bias: true │  │  │
│  │    • Linear  │       │     ┌─────┐  │  │  │            │  │  │
│  │    • FusedQKV│       └────▶│ W_v │──┤  │  │ [Infer]    │  │  │
│  │              │             └─────┘  │  │  └────────────┘  │  │
│  │  ▼ Templates │                     ▼  │                   │  │
│  │    • Transformer                ┌─────┐│  SHAPES          │  │
│  │    • MLP     │                  │ SDPA ││  ──────────────  │  │
│  │    • DBA Block                  └─────┘│  in:  [B,T,2048] │  │
│  │    • GQA Block                    │    │  out: [B,T,2048] │  │
│  │              │                    ▼    │                   │  │
│  │              │                ┌─────┐  │  FLOPS: 12.4M    │  │
│  │              │                │ out │  │  Params: 4.2M    │  │
│  │              │                └─────┘  │                   │  │
│  └──────────────┴──────────────────────────┴──────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  VALIDATION                                                │  │
│  │  ✓ Graph connected    ✓ All required inputs provided       │  │
│  │  ✓ Shapes compatible  ✓ W_q.d_out = n_heads × d_head       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Operation Picker

The left panel fetches operation schemas from the backend API (`/api/operations`). Operations are grouped by category (attention, projection, normalization, etc.) and can be dragged onto the canvas or double-clicked to insert.

### Canvas

Nodes are draggable. Edges are drawn between ports. Ports are typed—connecting incompatible types highlights the edge in red. The canvas supports:

- Zoom and pan
- Multi-select and box selection
- Keyboard shortcuts
- Undo/redo

### Inspector

Clicking a node opens the inspector panel showing:
- Configurable parameters with their types and defaults
- Inferred input/output shapes (propagated from the graph root)
- Parameter count and FLOP estimate

### Template Blocks

Pre-wired subgraphs appear in the operations panel under "Templates." Dragging a template onto the canvas inserts a collapsed block node. Blocks can be expanded to inspect internals or configured via exposed variables.

```
Collapsed:                    Expanded:
┌─────────────────┐           ┌─────────┐   ┌─────┐
│                 │           │   W_q   │──▶│     │
│  "Attention"    │    ═══▶   │   W_k   │──▶│SDPA │──▶
│  (12 nodes)     │           │   W_v   │──▶│     │
│  [expand]       │           └─────────┘   └─────┘
└─────────────────┘
```

### YAML Export

The canvas serializes to a YAML manifest via the backend API. The round-trip is lossless—loading a saved manifest restores the exact graph layout.

---

## The Research Microscope

Real-time visualization of every weight update is physically impractical—the data volume would overwhelm any channel. Instead, the frontend operates as a **microscope**: you focus on what matters, at the zoom level where questions become answerable.

### Zoom Levels

```
Model          → Loss, perplexity, throughput, total gradient norm
    ↓
Layer          → Per-layer gradient norms, activation statistics
    ↓
Component      → Attention head patterns, FFN activation distributions
    ↓
Weight         → Specific parameter matrices, their distributions
    ↓
Token          → How a specific input flows through the network
```

### Three Data Modes

**Continuous** (lightweight, always on):
- Loss, learning rate, gradient norm, throughput, step counter
- ~100 bytes/step via SSE to the frontend
- Powers the "training is alive" heartbeat

**Checkpointed** (rich snapshots at validation points):
- Attention patterns for fixed eval examples
- Activation statistics per layer
- Weight histograms, per-layer gradient norm history
- Eval set predictions
- Captured at checkpoint time, stored, queryable later

**Interactive** (on-demand probing):
- "Run this input through checkpoint X, show attention"
- "What's the gradient of output Y w.r.t. input Z?"
- "Compute saliency map for this prediction"
- Request/response; results cached

---

## Visualization Tools

### Training Dynamics

```
Loss ───────────────────────────────────────────
3.5 │ ╲
    │  ╲
3.0 │   ╲__
    │      ╲___
2.5 │          ╲____
    │               ╲_______
2.0 │                       ╲_______________
    └─────────────────────────────────────────
      0        2k        4k        6k        8k

Gradient Norm by Layer ─────────────────────────
Layer 12 │ ████████████████████▒▒▒▒░░░░
Layer 11 │ █████████████████████▒▒▒░░░░
...
Layer 0  │ █████████████████████████▒▒▒
```

### Attention Inspector

Click any attention node in the graph view to open the attention inspector for a specific input:

```
Layer 6, Head 3

         The  capital  of  France  is  [MASK]
The       ░░    ░░     ░░    ░░   ░░    ░░
capital   ▒▒    ██     ░░    ▒▒   ░░    ░░
of        ░░    ▒▒     ██    ▒▒   ░░    ░░
France    ░░    ▒▒     ▒▒    ██   ░░    ▒▒
is        ░░    ░░     ░░    ▒▒   ██    ▒▒
[MASK]    ░░    ▒▒     ░░    ██   ▒▒    ██

[Layer ◀ 6 ▶]  [Head ◀ 3 ▶]  [Compare to checkpoint...]
```

### Checkpoint Diff

Compare two checkpoints side by side:

```
Module                        │ Δ Norm  │ Δ Mean  │ % Changed
──────────────────────────────┼─────────┼─────────┼───────────
layers.0.attention.W_q        │ 0.142   │ +0.003  │ ██████ 8.2%
layers.0.attention.W_k        │ 0.138   │ +0.002  │ █████▓ 7.9%
layers.1.ffn.W_1              │ 0.127   │ -0.001  │ █████░ 7.1%
...
layers.11.attention.W_o       │ 0.012   │ +0.000  │ ▓░░░░░ 0.7%
```

### Anomaly Debugger

When the Notary detects anomalies during training:

```
⚠ ANOMALY DETECTED at step 7,342

Loss spiked: 2.3 → 847.2 (368×)

Gradient Analysis:
Layer 11 attention.W_o:  ████████████████████████ 1.2e+8 (!)
Layer 11 attention.W_q:  ██████████████████████   9.4e+7
Layer 11 ffn.W_1:        █████████               2.1e+4
Layer 10 attention.W_o:  ███                     1.2e+2

Likely cause: gradient explosion in Layer 11 attention output projection

[Inspect batch] [Compare to step 7341] [View checkpoint before spike]
```

### Training Replay

Scrub through training like a video. For any step, see:
- Attention patterns for a fixed probe input
- Top predictions with probabilities
- Loss, gradient norm, learning rate

---

## Semantic Zoom (Graph View)

The graph view applies semantic zoom to make large models navigable:

```
LEVEL 0: MODULE GRAPH (~50-200 nodes)
────────────────────────────────────
Load model → extract hierarchy → graph

         ┌──────────┐
         │ Embedding │
         └──────┬───┘
                │
         ┌──────┴───┐
    ┌────│  Layer 0 │────┐
    ▼    └──────────┘    ▼
┌───────┐           ┌───────┐
│ Attn  │           │  FFN  │
└───────┘           └───────┘
    │                   │
    └─────────┬─────────┘
         ┌────┴─────┐
         │  Layer 1 │
         └──────────┘
              ...

Click a node to drill down ▼

LEVEL 1: COMPONENT GRAPH (~10-50 nodes)
────────────────────────────────────
Click "Attention" → see internal ops

     ┌───────┐
     │   x   │ input
     └───┬───┘
         │
  ┌──────┼──────┐
  ▼      ▼      ▼
┌───┐  ┌───┐  ┌───┐
│W_q│  │W_k│  │W_v│ projections
└─┬─┘  └─┬─┘  └─┬─┘
  │      │      │
  ...   ...    ...
         ▼
     ┌───────┐
     │  out  │
     └───────┘

Click "W_q" to drill down ▼

LEVEL 2: WEIGHT INSPECTOR (panel view)
────────────────────────────────────
W_q: Linear(2048, 2048)
Parameters: 4,194,304

Distribution: mean=0.0001  std=0.023  sparsity=0.1%
[Weight heatmap] [Gradient history] [Compare to checkpoint]
```

Node color encodes the selected metric (gradient norm, activation magnitude, weight norm). Node size encodes parameter count.

---

## Routes

| Route                   | Description                                   |
|-------------------------|-----------------------------------------------|
| `/`                     | Dashboard                                     |
| `/research`             | Active experiments and runs                   |
| `/project`              | Project management                            |
| `/project/:id/edit`     | Architecture editor                           |
| `/kanban`               | Research task board                           |
| `/paper`                | Paper/report generator                        |
| `/docs`                 | Embedded documentation                        |

---

## API Endpoints

The frontend communicates with the Go backend over REST:

| Method | Path                         | Description                                     |
|--------|------------------------------|-------------------------------------------------|
| GET    | `/api/operations`            | All operation schemas (for op picker)           |
| GET    | `/api/blocks`                | All block schemas (for template panel)          |
| GET    | `/api/models`                | Available model templates                       |
| POST   | `/api/manifest/save`         | Save architecture as YAML manifest              |
| GET    | `/api/manifest/:id`          | Load a saved manifest                           |
| GET    | `/api/metrics/stream`        | SSE stream of live training metrics             |
| POST   | `/api/probe`                 | On-demand activation/attention probe            |
| GET    | `/api/checkpoint/:id/diff`   | Checkpoint diff                                 |
