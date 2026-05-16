# 🌈 caramba

**A substrate for A.I. research.**

Aiming to be an end-to-end research stack, caramba does everything in its power to let the researcher focus on their ideas, without having to worry about having to micro-manage the process.

Out of the box, caramba provides you with the tools to quickly iterate on ideas, and the ability to deeply inspect results.

---

## ✨ Features

- [x] Compute Primitives
  - [x] Activation (ReLU, LeakyReLU, SELU, Sigmoid, Tanh, GeLU, Swish, SwiGLU)
  - [x] Attention (SDPA, MQA, GQA, sliding window, softmax)
  - [x] Convolution (Conv1D, Conv2D, Conv3D, ConvTranspose2D)
  - [x] Embedding (token, RoPE, ALiBi, tied)
  - [x] Math (matmul, add/mul, exp/log, rmsnorm, layernorm, groupnorm, softmax, logsumexp, dropout)
  - [x] Pooling (avg, max, adaptive avg, adaptive max)
  - [x] Projection (linear, fused QKV, tied embedding)
  - [x] Shape (reshape, transpose, concat, split, view_as_heads, merge_heads, last_token, nearest upsample)
  - [x] Masking (causal mask, apply mask)
  - [x] Active Inference (free energy, expected free energy, belief update, precision weighting)
  - [x] Causal Inference (do-calculus, backdoor, frontdoor, CATE, IV, counterfactual, DAG factorization)
  - [x] Hawkes Process (intensity, kernel matrix, simulate, log-likelihood)
  - [x] Markov Blanket (partition, mutual information, internal/active flow)
  - [x] Predictive Coding (prediction, prediction error, representation/weight updates)
  - [x] VSA (bind, bundle, permute, inverse permute, similarity)
- [x] Multiple Compute Backends
  - [x] CPU (Go native)
  - [x] SIMD/Assembly
    - [x] AVX2 (amd64)
    - [x] SSE2 (amd64)
    - [x] NEON (arm64)
  - [x] CUDA
  - [x] METAL
  - [x] XLA
- [x] Optimizers (SGD, Adam, AdamW, AdaMax, AdaGrad, AdaDelta, RMSProp, Lion, LARS, LAMB, L-BFGS, Hebbian)
- [ ] Training Models
- [ ] Fine-tuning Models
- [x] Manifest Compiler (verify, canonicalize, CSE, algebraic simplify, fusion, DCE, memory planning, cost scheduling)
- [x] Hugging Face Hub Asset Resolver (revision-pinned, content-addressed, Xet CAS)
- [x] Provenance Ledger (signed by `pkg/notary`)
- [x] Streaming Chat Runtime (KV cache, sampling, `qpool` startup events)
- [x] Diffusion Pipeline (FlowMatch Euler, prompt encoder + denoiser + VAE decoder)
- [ ] Supported Pre-trained Models
  - [x] [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
  - [x] [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (Gated model, requires request for access)
  - [ ] [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)
  - [x] [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
  - [ ] [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)
  - [ ] [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)
  - [ ] [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
  - [ ] [stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
  - [ ] [meituan-longcat/LongCat-AudioDiT-3.5B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B)
  - [ ] [facebook/ijepa_vith16_1k](https://huggingface.co/facebook/ijepa_vith16_1k)
  - [ ] [facebook/vjepa2-vitg-fpc64-256](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256)
- [ ] Visual Node-Graph Architecture Builder
- [ ] ModelScope deep inspection tools
- [ ] Layer Surgery tools
- [ ] Hyperparameter Tuner
- [ ] Distributed Training
- [ ] Integrated Benchmarking Suite
- [ ] Deeply Integrated A.I. Assistant and Research Team
- [ ] Ergonomic WYSIWYG LaTeX Paper Editor
- [ ] Multi-User/Team Collaboration

## A manifest is a model

Architectures are declared, not coded. Here is a slice of Llama 3.2 1B Instruct as caramba sees it:

```yaml
name: Llama 3.2 1B Instruct

system:
  runtime:
    type: model
    backend: metal
    model:     { source: meta-llama/Llama-3.2-1B-Instruct }
    tokenizer: { source: meta-llama/Llama-3.2-1B-Instruct }
    generation:
      max_new_tokens: 256
      temperature: 0.8
      top_k: 50
      top_p: 0.95
      prompt_template: |+
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Cutting Knowledge Date: December 2023
        Today Date: 26 Jul 2024

        <|eot_id|><|start_header_id|>user<|end_header_id|>

        {{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


  topology:
    nodes:
      - id: embed_tokens
        op: embedding.token
        in:  [input_ids]
        out: [h_0]
        config: { vocab_size: 128256, d_model: 2048 }

      - id: transformer_layers
        op: control.repeat
        in:  [h_0]
        out: [h_16]
        repeat: 16
        index: i
        template:
          - { id: norm_${i},    op: math.rmsnorm,        in: [h_${i}],          out: [n_${i}],   config: { eps: 1e-05 } }
          - { id: q_${i},       op: projection.linear,   in: [n_${i}],          out: [q_${i}],   config: { in_features: 2048, out_features: 2048 } }
          - { id: k_${i},       op: projection.linear,   in: [n_${i}],          out: [k_${i}],   config: { in_features: 2048, out_features: 512  } }
          - { id: v_${i},       op: projection.linear,   in: [n_${i}],          out: [v_${i}],   config: { in_features: 2048, out_features: 512  } }
          - { id: qh_${i},      op: shape.view_as_heads, in: [q_${i}],          out: [qh_${i}],  config: { num_heads: 32 } }
          - { id: kh_${i},      op: shape.view_as_heads, in: [k_${i}],          out: [kh_${i}],  config: { num_heads: 8  } }
          - { id: vh_${i},      op: shape.view_as_heads, in: [v_${i}],          out: [vh_${i}],  config: { num_heads: 8  } }
          - id: rope_q_${i}
            op: positional.rope
            in:  [qh_${i}]
            out: [qr_${i}]
            config:
              base: 500000.0
              head_dim: 64
              mode: half
              rope_type: llama3
              rope_factor: 32.0
              rope_low_freq_factor: 1.0
              rope_high_freq_factor: 4.0
              rope_original_context: 8192
          - id: rope_k_${i}
            op: positional.rope
            in:  [kh_${i}]
            out: [kr_${i}]
            config:
              base: 500000.0
              head_dim: 64
              mode: half
              rope_type: llama3
              rope_factor: 32.0
              rope_low_freq_factor: 1.0
              rope_high_freq_factor: 4.0
              rope_original_context: 8192
          - { id: attn_${i},    op: attention.gqa,       in: [qr_${i}, kr_${i}, vh_${i}], out: [ah_${i}], config: { num_heads: 32, num_kv_heads: 8, head_dim: 64, causal: true } }
          - { id: merge_${i},   op: shape.merge_heads,   in: [ah_${i}],         out: [a_${i}] }
          # ... swiglu MLP, residuals, etc.

      - id: lm_head
        op: projection.linear
        in:  [final_norm]
        out: [logits]
        config: { in_features: 2048, out_features: 128256 }
```

Run it:

```bash
caramba chat --manifest model/llm/llama-3-2-1b-instruct.yml
```

caramba pulls the weights from the Hugging Face Hub (revision-pinned, cached, content-addressed), binds the safetensors into the named graph nodes by structure, lowers the graph to your selected backend, and streams tokens with a KV cache that lives in resident GPU memory.
Startup publishes `qpool` progress events for manifest resolution, backend selection, tokenizer loading, SafeTensors resolution, and runtime readiness, so long first-run Hub downloads do not look like a dead terminal.
The CLI subscribes to the same `qpool` broadcast stream and renders terminal progress bars for Hub transfers, SafeTensors loading, token generation, and diffusion denoising. While that TUI is active, standard JSON logging is suppressed and the latest `qpool` event is rendered as the status line above the progress bar.
Application-level concurrency is scheduled through `qpool`; goroutine creation is isolated inside the pool internals so downloads, long-running services, and background work share the same telemetry, cancellation, and backpressure path.

You did not write a model class. You did not write a forward pass. You described an architecture, and the substrate ran it.

Diffusion models use the same manifest path. The FLUX.2 Klein 4B denoiser
template lives at `model/diffusion/flux-2-klein-4b.yml`; its Qwen3 prompt
encoder lives at `model/diffusion/flux-2-klein-4b-text-encoder.yml`; its VAE
decoder lives at `model/diffusion/flux-2-klein-4b-vae-decoder.yml`. The image
command resolves those assets from the manifest, encodes the prompt, runs the
FlowMatch Euler denoising loop, decodes the latents through the manifest-backed
VAE when configured, and writes an RGB PNG:

The FLUX.2 binding path follows the checkpoint structure directly: affine-free
RMSNorm nodes are declared in the manifest, Diffusers Linear tensors are bound
with PyTorch row-major orientation, and the single-stream packed QKV/MLP plus
split `to_out` weights are sliced into the manifest graph without introducing a
model-specific runtime.

```bash
caramba image --manifest model/diffusion/flux-2-klein-4b.yml "a brass observatory on a storm cliff"
```

---

## What you get

**One IR, four resident backends.** Every operation has a Go scalar reference, and accelerator graphs are only legal when the selected backend advertises a resident kernel for that operation:

- **CPU** with hand-written AVX2, SSE2, and NEON assembly — each ISA has its own kernel, no scalar branches in disguise.
- **CUDA** native `.cu` kernels for activations, attention, convolution, embeddings, causal masking, pooling.
- **Metal** native `.metal` shaders with resident KV caches and on-device tensor lifecycle.
- **XLA** via PJRT — your graph becomes a StableHLO module and runs through whichever plugin you point caramba at.

**A real operation library.** Standard primitives — attention (SDPA, GQA, flash, causal), activations (GeLU exact, SwiGLU, Mish, …), normalization, convolution, pooling, projection, embedding (token, RoPE, ALiBi, sinusoidal), shape transforms and nearest-neighbor upsample — plus a set of operations you will not find in PyTorch: active inference, Hawkes processes, Markov blanket detection, predictive coding, vector symbolic architectures. Optimizers include SGD, Adam(W), Lion, LARS, LAMB, L-BFGS, each with native kernels on every backend. Diffusion decoder support now binds VAE convolution and GroupNorm tensors through the same SafeTensors-to-IR path.

**A compiler, not an interpreter.** Manifests pass through `pkg/manifest`'s pipeline: verification, canonicalization, semantic CSE, algebraic simplification, legality-aware fusion, side-effect-aware DCE, memory planning, and cost scheduling — before lowering. Fusions are declared and validated against per-backend capability contracts, not invented at runtime.

**A visual editor.** The frontend (`frontend/`) is a Vite + React + Flume canvas that reads the operation registry from the backend, supports template blocks (`TransformerBlock`, `MLP`, `DBA`, `GQA`, …), and serializes back to the same YAML the CLI consumes. Dragging nodes and editing text are two paths to the same artifact.

**Hub-native asset resolution.** `pkg/hub` provides a revision-aware local cache that mirrors `hf_hub_download` / `snapshot_download` semantics: refs or cached `info/<revision>` metadata, commit-pinned snapshots, content-addressed blobs, per-file metadata, Xet CAS reconstruction. The default cache is `${HOME}/.cache/huggingface/hub`, so existing Hugging Face snapshots are checked before caramba downloads. Manifests reference assets by plain ID (`openai-community/gpt2`) or by explicit locator (`hf://model/openai-community/gpt2@main`).

**Provenance that travels with the artifact.** A trained model is its weights *and* the graph that produced them *and* a ledger entry signed by `pkg/notary`. Share a model and you share its history.

---

## Quick start

```bash
go install github.com/theapemachine/caramba@latest

caramba serve                                                    # HTTP API
caramba chat                                                     # terminal chat against the default manifest
caramba chat --manifest model/llm/llama-3-2-1b-instruct.yml      # pick a different one
caramba research my-ablation-study                               # scaffold a study
```

`caramba research <name>` lays out a project as a directory under version control:

```
research/project/my-ablation-study/
├── manifest/
│   ├── architecture/   # the architectures under comparison
│   └── operation/      # custom operations specific to this study
└── paper/              # write-up
```

Configuration lives in [cmd/asset/config.yml](cmd/asset/config.yml) and is loaded through [pkg/config](pkg/config). The resolver tries `--config`, then `./cmd/asset/config.yml`, `./config.yml`, `$HOME/.caramba/config.yml`, and finally the binary's embedded default.

→ [Getting Started](./docs/getting-started.md)

---

## Building for a specific backend

```bash
# CPU — Go + AVX2/SSE2/NEON. Always available.
go build ./pkg/backend/compute/cpu/...

# CUDA — Linux, NVIDIA CUDA toolkit
CGO_ENABLED=1 go build -tags "cgo cuda" ./pkg/backend/compute/cuda/...

# Metal — macOS, Xcode command-line tools (darwin + cgo selects Metal automatically)
CGO_ENABLED=1 go build -tags cgo ./pkg/backend/compute/metal/...

# XLA via PJRT — configure compute.xla in cmd/asset/config.yml first
go build -tags "cgo xla" ./pkg/backend/compute/xla/...
```

Every backend implements the same interface, and the optimizer does the same math everywhere:

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

Backend kernels upload values once into a resident tensor store and only download at real boundaries. The executor releases owned dependencies after their last graph consumer; the host arena reuses released spans. Host-staged dispatch is restricted to the host backend, so Metal, CUDA, and XLA paths cannot silently route through CPU slices.

→ [Compute Backends](./docs/compute.md)

---

## Operations at a glance

| Category          | Examples                                            |
|-------------------|-----------------------------------------------------|
| Activation        | ReLU, GeLU (exact erf), SwiGLU, Tanh, Mish          |
| Attention         | SDPA, DBA, GQA, Multi-head, Causal, Flash           |
| Embedding         | Token, Positional (RoPE, ALiBi, sinusoidal)         |
| Normalization     | LayerNorm, RMSNorm, GroupNorm, BatchNorm            |
| Projection        | Linear, FusedQKV, LoRA                              |
| Convolution       | Conv1D, Conv2D, depthwise, grouped, transposed      |
| Shape             | Reshape, transpose, concat, split, nearest upsample |
| Pooling           | Mean, Max, Attention pooling                        |
| Active Inference  | Free energy minimization, precision weighting       |
| Causal            | Temporal difference, causal intervention            |
| Hawkes Process    | Point process attention kernels                     |
| Markov Blanket    | Blanket detection, free energy decomposition        |
| Predictive Coding | Hierarchical prediction error                       |
| VSA               | Hyperdimensional binding, bundling, cleanup memory  |
| Optimizers        | SGD, Adam(W), Lion, LARS, LAMB, L-BFGS              |

If a kernel is missing for a backend, lowering or execution fails loudly at that backend boundary. There is no silent fallback.

→ [Operations](./docs/operations.md)

---

## Repository layout

```
cmd/                Cobra CLI: serve, chat, research
  asset/config.yml  The single config source
pkg/
  manifest/         YAML → IR compiler, registry, lowering
  backend/
    compute/        Runner interface + cpu/, cuda/, metal/, xla/
    api/            HTTP server
  chat/             Streaming chat session, sampling, KV cache
  hub/              Hugging Face cache, Xet CAS
  tokenizer/        ByteLevel BPE
  model/            Weight binding, SafeTensors loader
  notary/           Identity + provenance ledger
  store/            S3, Elasticsearch, Neo4j, Qdrant, DeepLake
  config/           Single config gateway
frontend/           Vite + React + Flume node editor
docs/               Long-form documentation
AGENTS.md           Backend implementation contract — required reading for kernel work
```

---

## Testing

Every code file has a `_test.go` mirror. Tests are GoConvey-style ("Given X, it should Y", nested). Backend kernels run parity tests against the scalar reference at `N ∈ {1, 7, 64, 1024, 8192}` with tight ULP bounds — the tolerance is a contract, not a knob.

```bash
go test ./...
CGO_ENABLED=1 go test -tags cgo            ./pkg/backend/compute/metal/...
CGO_ENABLED=1 go test -tags "cgo cuda"     ./pkg/backend/compute/cuda/...
              go test -tags "cgo xla"      ./pkg/backend/compute/xla/...
```

---

## Documentation

| Document                                       | What's inside                                       |
|------------------------------------------------|-----------------------------------------------------|
| [Getting Started](./docs/getting-started.md)   | Install, first chat, first study                    |
| [Architecture](./docs/architecture.md)         | System design, IR, executor                         |
| [Manifest & Governance](./docs/manifest.md)    | Manifest grammar, compiler pipeline, examples       |
| [Compute Backends](./docs/compute.md)          | CPU/SIMD, CUDA, Metal, XLA in depth                 |
| [Operations](./docs/operations.md)             | Operation library, SIMD kernels, custom ops         |
| [Frontend & Visualization](./docs/frontend.md) | Node editor, microscope tooling                     |
| [The Notary](./docs/notary.md)                 | Identity, ledger, custody model                     |
| [Agents](./docs/agents.md)                     | Conversational ingress, LLM providers               |
| [AGENTS.md](./AGENTS.md)                       | Backend implementation contract for contributors    |

---

## License

MIT

---

<p align="center">
  <i>The model knows how it was made.</i>
</p>
