# Getting Started

---

## Prerequisites

- Go 1.26 or later
- For the frontend: Node.js 20+
- For CUDA: Linux, NVIDIA CUDA toolkit
- For Metal: macOS, Xcode Command Line Tools
- For XLA: XLA headers and a PJRT plugin library

---

## Installation

```bash
git clone https://github.com/theapemachine/caramba
cd caramba

# Build the CLI
go build -o caramba .

# Or install globally
go install .
```

---

## Running the Server

The server hosts the API and serves the frontend:

```bash
caramba serve
```

The frontend is available at `http://localhost:3000` (or whichever port is configured). The API is available at `http://localhost:8080/api`.

Configuration is loaded from `cmd/asset/config.yml` through `pkg/config`; edit that file or pass `--config` to use an alternate YAML file.

---

## Starting a Research Project

Scaffold a new project with:

```bash
caramba research <project-name>
```

This creates:

```
research/project/<project-name>/
├── manifest/
│   ├── master.yml          # Top-level project manifest
│   ├── architecture/       # Architecture manifests
│   └── operation/          # Custom operation definitions
└── paper/                  # LaTeX paper artifacts
```

The generated `master.yml` is a starting point. Edit it to declare your intent.

---

## Writing a Manifest

A minimal manifest that trains a two-layer transformer:

```yaml
name: my_first_experiment
description: Small transformer on wikitext

variables:
  d_model: 256
  n_heads: 4
  vocab_size: 50257

datasets:
  train:
    source: s3://my-bucket/wikitext-train
    format: arrow
  eval:
    source: s3://my-bucket/wikitext-val
    format: arrow

trainer:
  optimizer: adam
  learning_rate: 3.0e-4
  max_steps: 5000
  warmup_steps: 500
  batch_size: 32

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
          vocab_size: ${vocab_size}
          d_model: ${d_model}

      - repeat: 2
        index: layer_idx
        template:
          - id: attn_${layer_idx}
            op: attention.sdpa
            in: [layer_${layer_idx}_in]
            out: [attn_${layer_idx}_out]
            config:
              d_model: ${d_model}
              n_heads: ${n_heads}
              causal: true
          - id: ffn_${layer_idx}
            op: projection.linear
            in: [attn_${layer_idx}_out]
            out: [layer_${next_layer_idx}_in]
            config:
              d_in: ${d_model}
              d_out: ${d_model}

      - id: lm_head
        op: projection.linear
        in: [layer_2_in]
        out: [logits]
        config:
          d_in: ${d_model}
          d_out: ${vocab_size}

    outputs: [logits]
```

---

## Using the Architecture Editor

Instead of writing manifests by hand, use the visual editor:

1. Open `http://localhost:3000` in the browser
2. Navigate to **Project → New** or open an existing project
3. Drag operations from the left panel onto the canvas
4. Connect ports to wire the computation graph
5. Click any node to configure its parameters in the inspector
6. Use the **Templates** section for pre-wired blocks (TransformerBlock, DBA, GQA, etc.)
7. Click **Export** to save as a YAML manifest

---

## Running an Experiment

Once you have a manifest:

```bash
# Validate the manifest
caramba manifest validate path/to/manifest.yml

# Run the experiment
caramba research run path/to/manifest.yml

# Check status
caramba research status

# List assets
caramba asset list
```

The experiment follows the validation flow:
1. Notary validates the manifest
2. Compiler lowers YAML to IR
3. Orchestrator optimizes the IR
4. Runner dispatches to hardware
5. Notary checkpoints at defined intervals
6. On completion: commit (verified) or void (failed)

---

## Choosing a Backend

By default, the CPU backend is used. To select a specific backend:

```bash
# CUDA (Linux only, CUDA toolkit required)
CGO_ENABLED=1 go run -tags "cgo cuda" . serve

# Metal (macOS only)
CGO_ENABLED=1 go run . serve

# XLA (configure compute.xla in cmd/asset/config.yml first)
CGO_ENABLED=1 go run -tags "cgo xla" . serve
```

See [Compute Backends](./compute.md) for full setup instructions.

---

## Running Tests

```bash
# All tests
go test ./...

# CPU compute tests
go test ./pkg/backend/compute/cpu/...

# Manifest compiler tests
go test ./pkg/manifest/...

# CUDA tests (Linux, CUDA toolkit required)
CGO_ENABLED=1 go test -tags "cgo cuda" ./pkg/backend/compute/cuda/...

# Metal tests (macOS)
CGO_ENABLED=1 go test ./pkg/backend/compute/metal/...

# XLA tests
CGO_ENABLED=1 go test -tags "cgo xla" ./pkg/backend/compute/xla/...
```

Tests use [Goconvey](https://github.com/smartystreets/goconvey) with a nested BDD structure (`Given` / `It should`). Every file has a corresponding `_test.go` file.

---

## Frontend Development

```bash
cd frontend
npm install
npm run dev
```

The frontend uses **TanStack Start + Router**. See [Frontend & Visualization](./frontend.md) for the full architecture.

---

## Next Steps

| Document                                     | What it covers                              |
|----------------------------------------------|---------------------------------------------|
| [Architecture](./architecture.md)            | System internals, IR, distributed model     |
| [Compute Backends](./compute.md)             | CPU/SIMD, CUDA, Metal, XLA                  |
| [Manifest & Governance](./manifest.md)       | Manifest schema, atomic intent, lifecycle   |
| [The Notary](./notary.md)                    | Validation, ledger, custody                 |
| [Operations](./operations.md)                | Full operation library reference            |
| [Frontend & Visualization](./frontend.md)    | Editor, microscope, graph view              |
| [Agents](./agents.md)                        | Research team interface, LLM providers      |
