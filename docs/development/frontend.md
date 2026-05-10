This document outlines the design for an ML training visualization system framed as a research **microscope**: it never tries to visualize every tensor at training speed, yet still supports deep, query-driven inspection of what the model is doing.

Real-time visualization of every weight update in a billion-parameter model remains physically impractical—the data volume would overwhelm any channel—but **microscopic** tooling can still expose what matters **on demand**, at the zoom level where questions become answerable.

But "microscope" remains the operative metaphor because it matches how investigators actually work:

1. **Zoom** to different magnifications
2. **Focus** on specific regions
3. **Use different stains** to reveal different structures
4. **Compare samples** side by side

The ML equivalent:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE RESEARCH MICROSCOPE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ZOOM LEVELS                                                               │
│   ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│   Model          │ Loss, perplexity, throughput, total gradient norm        │
│       ↓          │                                                          │
│   Layer          │ Per-layer gradient norms, activation statistics          │
│       ↓          │                                                          │
│   Component      │ Attention head patterns, FFN activation distributions    │
│       ↓          │                                                          │
│   Weight         │ Specific parameter matrices, their distributions         │
│       ↓          │                                                          │
│   Token          │ How a specific input flows through the network           │
│                                                                             │
│                                                                             │
│   "STAINS" (views that reveal different structures)                         │
│   ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│   • Gradient Flow    — Where are gradients flowing/vanishing?               │
│   • Activation Map   — What's lighting up for this input?                   │
│   • Attention Focus  — Where is the model looking?                          │
│   • Learning Velocity— What's changing fastest right now?                   │
│   • Dead Neurons     — What's never activating?                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Key Insight: Query-Driven, Not Stream-Driven

Instead of "here's a dashboard streaming everything," the paradigm is:

**"Ask questions, get visualizations"**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CONTINUOUS (lightweight stream, always on)                                │
│   ─────────────────────────────────────────────────────────────────────     │
│   • Loss                          │                                         │
│   • Learning rate                 │  ~100 bytes/step                        │
│   • Gradient norm (scalar)        │  SSE to frontend                        │
│   • Throughput (tokens/sec)       │                                         │
│   • Step counter                  │                                         │
│                                                                             │
│   This powers the "training is alive" heartbeat.                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CHECKPOINTED (rich snapshots at validation points)                        │
│   ─────────────────────────────────────────────────────────────────────     │
│   • Attention patterns for fixed eval examples                              │
│   • Activation statistics per layer                                         │
│   • Weight histograms                                                       │
│   • Per-layer gradient norms history                                        │
│   • Eval set predictions                                                    │
│                                                                             │
│   Captured at checkpoint time. Stored to disk/S3. Queryable later.          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INTERACTIVE (on-demand probing)                                           │
│   ─────────────────────────────────────────────────────────────────────     │
│   • "Run this input through checkpoint X, show attention"                   │
│   • "What's the gradient of output Y w.r.t. input Z?"                       │
│   • "Compute saliency map for this prediction"                              │
│   • "What tokens most influenced this output?"                              │
│                                                                             │
│   Request/response. Computed when asked. Results cached.                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   COMPARATIVE (analysis across checkpoints/architectures)                   │
│   ─────────────────────────────────────────────────────────────────────     │
│   • Diff two checkpoints                                                    │
│   • Compare attention patterns: baseline vs DBA                             │
│   • Loss curves overlaid                                                    │
│   • "What changed between step 5000 and step 10000?"                        │
│                                                                             │
│   Computed on request. Results become artifacts.                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Concrete Visualizations That Would Be Valuable

### 1. Training Dynamics View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING DYNAMICS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Loss ───────────────────────────────────────────────────────────────────  │
│   3.5 │ ╲                                                                   │
│       │  ╲                                                                  │
│   3.0 │   ╲__                                                               │
│       │      ╲___                                                           │
│   2.5 │          ╲____                                                      │
│       │               ╲_______                                              │
│   2.0 │                       ╲_______________                              │
│       └─────────────────────────────────────────────────────────────────    │
│         0        2k        4k        6k        8k        10k                │
│                                                                             │
│   Gradient Norm by Layer ─────────────────────────────────────────────────  │
│                                                                             │
│   Layer 12 │ ████████████████████▒▒▒▒▒▒▒▒▒░░░░░░░░░                        │
│   Layer 11 │ █████████████████████▒▒▒▒▒▒▒▒░░░░░░░░                         │
│   Layer 10 │ ██████████████████████▒▒▒▒▒▒▒░░░░░░░                          │
│   ...                                                                       │
│   Layer 1  │ ████████████████████████████▒▒▒▒▒▒                            │
│   Layer 0  │ █████████████████████████████▒▒▒▒▒                            │
│            └────────────────────────────────────                            │
│              0.0     0.5     1.0     1.5     2.0  (gradient norm)          │
│                                                                             │
│   ◉ Now  ○ Step 5000  ○ Step 2500  ○ Step 0                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Attention Inspector (Interactive, On-Demand)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ATTENTION INSPECTOR                              Checkpoint: step_10000    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: "The capital of France is [MASK]"                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Layer 6, Head 3                                                    │   │
│  │                                                                     │   │
│  │          The   capital   of    France   is    [MASK]               │   │
│  │  The      ░░    ░░       ░░      ░░     ░░      ░░                 │   │
│  │  capital  ▒▒    ██       ░░      ▒▒     ░░      ░░                 │   │
│  │  of       ░░    ▒▒       ██      ▒▒     ░░      ░░                 │   │
│  │  France   ░░    ▒▒       ▒▒      ██     ░░      ▒▒                 │   │
│  │  is       ░░    ░░       ░░      ▒▒     ██      ▒▒                 │   │
│  │  [MASK]   ░░    ▒▒       ░░      ██     ▒▒      ██                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Layer ◀ 6 ▶]  [Head ◀ 3 ▶]  [Compare to checkpoint...]                   │
│                                                                             │
│  Observation: Head 3 at layer 6 strongly attends "France" → "[MASK]"        │
│               This is the "fact retrieval" head.                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Architecture Comparison View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE COMPARISON                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Comparing: baseline_s1337 vs dba_s1337                                     │
│  Eval set: wikitext_val (500 examples)                                      │
│                                                                             │
│  ┌────────────────────────────┐  ┌────────────────────────────┐            │
│  │      BASELINE              │  │        DBA                 │            │
│  │                            │  │                            │            │
│  │  Perplexity: 24.3          │  │  Perplexity: 23.8 (↓2.1%)  │            │
│  │  Latency:    142ms/batch   │  │  Latency:    98ms (↓31%)   │            │
│  │  Memory:     12.4 GB       │  │  Memory:     8.2 GB (↓34%) │            │
│  │  Params:     1.2B          │  │  Params:     1.1B (↓8%)    │            │
│  │                            │  │                            │            │
│  └────────────────────────────┘  └────────────────────────────┘            │
│                                                                             │
│  Attention Pattern Comparison (same input: "The quick brown fox")           │
│  ┌────────────────────────────┐  ┌────────────────────────────┐            │
│  │  ▓▓░░░░░░░░                │  │  ██▒▒░░░░░░                │            │
│  │  ░░▓▓░░░░░░                │  │  ▒▒██░░░░░░                │            │
│  │  ░░░░▓▓▒▒░░  (diffuse)     │  │  ░░░░██░░░░  (sharper)     │            │
│  │  ░░░░▒▒▓▓▒▒                │  │  ░░░░▒▒██░░                │            │
│  │  ░░░░░░▒▒▓▓                │  │  ░░░░░░░░██                │            │
│  └────────────────────────────┘  └────────────────────────────┘            │
│                                                                             │
│  DBA shows sharper attention patterns with 31% lower latency.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. The "Debugger" View (for when things go wrong)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚠ ANOMALY DETECTED at step 7,342                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Loss spiked: 2.3 → 847.2 (368x)                                           │
│                                                                             │
│  Gradient Analysis:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Layer 11 attention.W_o:  ████████████████████████ 1.2e+8 (!)       │   │
│  │  Layer 11 attention.W_q:  ██████████████████████   9.4e+7           │   │
│  │  Layer 11 ffn.W_1:        █████████               2.1e+4            │   │
│  │  Layer 10 attention.W_o:  ███                     1.2e+2            │   │
│  │  ...                                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Likely cause: Gradient explosion in Layer 11 attention output projection  │
│                                                                             │
│  Batch that triggered this:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tokens: ["<|endoftext|>", "<|endoftext|>", "<|endoftext|>", ...]   │   │
│  │  Source: fineweb_1b.npy, indices [7342000:7342016]                  │   │
│  │  Suspicion: Degenerate batch with repeated special tokens           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Inspect batch] [Compare to step 7341] [View checkpoint before spike]      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│   │  Live View    │  │   Snapshot    │  │  Comparison   │                  │
│   │  (SSE stream) │  │   Browser     │  │  Workbench    │                  │
│   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                  │
│           │                  │                  │                           │
│           │                  │                  │                           │
│           ▼                  ▼                  ▼                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Query Interface                                │  │
│   │                                                                     │  │
│   │  "Show attention for input X at checkpoint Y"                       │  │
│   │  "Compare gradient norms between baseline and DBA"                  │  │
│   │  "What was the batch at step 7342?"                                 │  │
│   │                                                                     │  │
│   └───────────────────────────────┬─────────────────────────────────────┘  │
│                                   │                                        │
└───────────────────────────────────┼────────────────────────────────────────┘
                                    │
                                    │ Cap'n Proto RPC
                                    │
┌───────────────────────────────────┼────────────────────────────────────────┐
│                                   │                                        │
│                              BACKEND                                       │
├───────────────────────────────────┼────────────────────────────────────────┤
│                                   ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     Analysis Actor                                  │  │
│   │                                                                     │  │
│   │  interface Analysis {                                               │  │
│   │    # Live stream (lightweight)                                      │  │
│   │    subscribe @0 (experiment :Experiment) -> (stream :MetricStream); │  │
│   │                                                                     │  │
│   │    # Snapshot queries                                               │  │
│   │    getAttention @1 (checkpoint :Ref, input :Tokens) -> (patterns);  │  │
│   │    getGradients @2 (checkpoint :Ref, layer :Text) -> (gradients);   │  │
│   │    getActivations @3 (checkpoint :Ref, input :Tokens) -> (acts);    │  │
│   │                                                                     │  │
│   │    # Comparison                                                     │  │
│   │    compare @4 (a :Ref, b :Ref, metric :Text) -> (comparison);       │  │
│   │                                                                     │  │
│   │    # Debugging                                                      │  │
│   │    inspectBatch @5 (experiment :Ref, step :UInt64) -> (batch);      │  │
│   │    findAnomalies @6 (experiment :Ref) -> (anomalies :List);         │  │
│   │  }                                                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                   │                                        │
│                                   ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     Storage / Model                                 │  │
│   │                                                                     │  │
│   │  Checkpoints with:                                                  │  │
│   │  - weights (always)                                                 │  │
│   │  - optimizer state (always)                                         │  │
│   │  - eval metrics (always)                                            │  │
│   │  - attention snapshots for fixed eval set (at checkpoints)          │  │
│   │  - activation statistics (at checkpoints)                           │  │
│   │  - batch indices (for debugging)                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Data We Need to Capture

At different times, capture different granularities:

| When | What | Size | Purpose |
|------|------|------|---------|
| Every step | loss, lr, grad_norm, throughput | ~100 bytes | Live heartbeat |
| Every N steps | per-layer grad norms | ~1KB | Training dynamics |
| At checkpoints | attention for fixed eval set | ~10-100MB | Deep analysis |
| At checkpoints | activation statistics per layer | ~1MB | Distribution tracking |
| At checkpoints | weight histograms | ~1MB | Weight evolution |
| On anomaly | full batch data, all gradients | ~100MB | Debugging |
| On demand | attention for arbitrary input | computed | Interactive probing |

## What Makes This a "Microscope"

1. **You don't see everything at once** — you focus on what matters
2. **Different magnifications** — model → layer → head → weight
3. **Different stains** — gradient view, attention view, activation view
4. **Comparison** — side-by-side, overlay, diff
5. **Recording** — capture a "session" and replay it
6. **Annotation** — mark interesting findings, link to paper

The real-time aspect is just the "heartbeat." The analysis is query-driven, computed on demand, and the results become part of the Model's audit trail.

### Next Steps / Implementation Roadmap

Adjacent to the microscope framing above (“What aspects would you want to prioritize first?”), tie work to sibling specs and phased delivery:

- **Specs & infra:** [`compiler.md`](./compiler.md), [`architecture.md`](./architecture.md), [`architecture-builder.md`](./architecture-builder.md), [`migration.md`](./migration.md).
- **Stack / runtime:** frontend uses **TanStack Start + Router** with **SSE** bridging to Cap’n-backed services (capture exact versions in whichever release checklist you attach to milestones).
- **CI / deployments:** derive from workspace root automation (reuse the canonical pipeline README once published); snapshot environment expectations alongside feature flags controlling experimental UI shells.
- **Tracking:** populate **Linear/Jira/epic IDs** on the canonical project board (placeholder until numbers exist: `TRACKER_BOARD_URL`).

**Prioritized checklist (handoff)**

1. PoC: live SSE heartbeat + timeline snapshot picker + anomaly drill-through path.
2. Required APIs surfaced in Cap’n-RPC contracts (streaming metrics, replay queries).
3. Instrumentation parity: trace IDs bridging UI actions ↔ backend executions.
4. UX flows anchored in trust states (void → inspect → revise manifest → rerun).

Does this framing help? What aspects would you want to prioritize first?