# 🌈 caramba

**A substrate for A.I. research.**

**Caramba** is a comprehensive machine learning research stack built to guide you through the entire lifecycle of A.I. development. Whether you want to rapidly prototype a new concept or dive deep into low-level hardware optimization, Caramba provides a dedicated environment tailored to your exact workflow. 

Operating on the core philosophy that **a manifest is a model**, Caramba allows you to define complex architectures entirely via YAML files instead of writing code. It acts as a complete laboratory, seamlessly taking you from your initial idea to a heavily optimized, fully profiled pipeline.

**Core Capabilities:**

* 📝 **Fully Manifest-Driven:** Declare your topology in simple YAML. Caramba goes far beyond standard layers and operations, allowing you to easily express sophisticated, non-standard mathematical primitives and advanced custom architectures.
* 🚀 **Flexibility & Optimization:** Dedicated to high performance, Caramba gives you the tools to choose your level of abstraction. Move fast to iterate on high-level ideas, or drop down for granular, low-level control over compute and memory.
* 🔬 **Sophisticated Inspection:** Deeply understand your network's behavior. Caramba is equipped with advanced inspection and profiling tooling, bringing clarity to every step of the end-to-end research process.
* 🤖 **Integrated A.I. Collaboration:** Supercharge your workflow with a built-in A.I. assistant and virtual research team. Caramba is built from the ground up to support both human team collaboration and agentic brainstorming.
* 🔐 **Zero-Compromise Privacy:** Working with proprietary or sensitive data? Caramba can operate entirely in an optional "local-only" mode, ensuring your research and intellectual property never leave your secure environment.

## ✨ Features

- [x] Compute Primitives
  - [x] Activation (ReLU, LeakyReLU, SELU, Sigmoid, Tanh, GeLU, Swish, SwiGLU)
  - [x] Attention (SDPA, MQA, GQA, sliding window, softmax)
  - [x] Convolution (Conv1D, Conv2D, Conv3D, ConvTranspose2D)
  - [x] Embedding (token, RoPE, ALiBi, tied)
  - [x] Math (matmul, add/mul, exp/log, rmsnorm, layernorm, groupnorm, softmax, logsumexp, dropout, sin, cos)
  - [x] Pooling (avg, max, adaptive avg, adaptive max)
  - [x] Projection (linear, fused QKV, tied embedding)
  - [x] Shape (reshape, transpose, concat, split, view_as_heads, merge_heads, last_token, nearest upsample)
  - [x] Masking (causal mask, apply mask)
  - [x] Active Inference (free energy, expected free energy, belief update, precision weighting)
  - [x] Energy-Based Model Blocks (Boltzmann distribution, EBM free energy, Langevin step, contrastive phase)
  - [x] Causal Inference (do-calculus, backdoor, frontdoor, CATE, IV, counterfactual, DAG factorization)
  - [x] Hawkes Process (intensity, kernel matrix, simulate, log-likelihood)
  - [x] Markov Blanket (partition, mutual information, internal/active flow)
  - [x] Predictive Coding (prediction, prediction error, representation/weight updates)
  - [x] VSA (bind, bundle, permute, inverse permute, similarity)
- [x] Multiple Compute Backends
  - [x] CPU (Go native)
  - [x] SIMD/Assembly
    - [ ] AVX-512 (amd64)
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
- [x] SafeTensors architecture manifests (`from_safetensors`, config-driven registry lookup, direct tensor binding)
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

## 🚀 Quick start

```bash
go install github.com/theapemachine/caramba@latest

caramba serve
```

The above command brings up just the HTTP API, which means you still have to bring up the data stores yourself.

Alternatively you could grab the `docker-compose.yml` from this repository to make this process much easier.

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

## Building for a specific backend

```bash
# CPU — Go + AVX2/SSE2/NEON. Always available.
go build ./pkg/backend/compute/cpu/...

# CUDA — Linux, NVIDIA CUDA toolkit
CGO_ENABLED=1 go build -tags "cgo cuda" ./pkg/backend/device/cuda/...

# Metal — macOS, Xcode command-line tools (darwin + cgo selects Metal automatically)
go generate ./pkg/backend/device/metal
CGO_ENABLED=1 go build -tags cgo ./pkg/backend/device/metal/...

# XLA via PJRT — configure compute.xla in cmd/asset/config.yml first
go build -tags "cgo xla" ./pkg/backend/device/xla/...
```

The Metal backend embeds `pkg/backend/device/metal/kernels.metallib`
from `pkg/backend/device/metal/*.metal`. Dense elementwise binary and
unary kernels cover `float32`, `float16`, and `bfloat16`, including
extended unary math and activation kernels for `rsqrt`, `exp`, `log`,
`sin`, `cos`, `tanh`, `sigmoid`, `silu`, `swish`, `softsign`, `elu`,
`selu`, `leaky_relu`, `hardsigmoid`, and `hardswish`; Metal shape
kernels cover concat/split/head reshape/last-token/transpose/upsample
movement across the same storage dtypes with dtype-specific shader
entry points and `uint4` movement for aligned contiguous ranges.
Metal matmul kernels cover `matmul` and fused `matmul_add` for the
same storage dtypes with tiled threadgroup execution. Metal projection
and model kernels cover `linear`, `fused_qkv`, `lora_merge`, and
`lora_apply` for the same storage dtypes; the LoRA apply path stages
the rank-space intermediate in device scratch storage and submits the
two GPU stages in one command buffer. Metal transformer support covers
`attention`, `flash_attention`, `multi_head_attention`,
`grouped_query_attention`, `sliding_window_attention`,
`embedding_lookup`, `embedding_bag`, `apply_mask`, `causal_mask`, and
`alibi_bias`, and `rope` for the same storage dtypes; attention uses tiled GPU
stages for score construction and weighted output with a
device-resident float32 score buffer, flash attention uses a
row/value-tile online softmax kernel with threadgroup dot-product
reduction, and the multi-head variants use row/head/value-tile kernels
with GQA KV-head sharing and causal sliding-window masks. RoPE rotates
consecutive head-dimension pairs directly on the device. Embedding
kernels report invalid index data through the asynchronous command
completion path.
Metal vision kernels cover `conv1d`, `conv2d`, `conv3d`,
`conv_transpose2d`, `max_pool2d`, `avg_pool2d`,
`adaptive_avg_pool2d`, and `adaptive_max_pool2d` for the same storage
dtypes using NCL/NCHW/NCDHW memory layout and GPU-side accumulation.
Metal optimizer kernels cover `adam_step`, `adamw_step`,
`adamax_step`, `adagrad_step`, `rmsprop_step`, `lion_step`,
`sgd_step`, `lars_step`, `lbfgs_step`, and `hebbian_step` with
reduced-dtype params/gradients/output and float32 optimizer state.
Metal quantization kernels cover `int8_dequant`, `int4_dequant`, and
`int8_quant` with default scale/zero-point semantics matching the
scalar registry. Metal loss kernels cover `mse_loss`, `mae_loss`,
`huber_loss`, `binary_cross_entropy`, `cross_entropy`, and
`kl_divergence` for `float32`, `float16`, and `bfloat16`; pair losses
run chunked partial reductions and cross-entropy runs one row-local
threadgroup per batch row before a device finalize stage. Metal
reduction kernels cover `sum`, `mean`, `prod`, `reduce_min`,
`reduce_max`, `argmin`, `argmax`, `l1_norm`, `l2_norm`, `variance`,
and `stddev` for the same storage dtypes with chunked partial
reductions and a device finalize stage. Metal math utility kernels
cover `inv_sqrt_dim_scale`, `logsumexp`, and `outer` for the same
storage dtypes; `logsumexp` uses row-local parallel max and sum
reductions over the trailing dimension.
Metal softmax covers the same storage dtypes with one threadgroup per
row, parallel max reduction, parallel sum reduction, and normalized
dtype-native writes. Metal normalization covers `layernorm` and
`rmsnorm` for the same storage dtypes with row-local reductions and
dtype-native writes. Metal research kernels cover `vsa_bind`,
`vsa_bundle`, `vsa_permute`, `vsa_inverse_permute`, `pc_prediction`,
`pc_prediction_error`, `pc_update_representation`, and
`pc_update_weights` for the same storage dtypes, using fp32
accumulation for predictive-coding matrix/vector contractions. Metal
active-inference kernels cover `free_energy`, `expected_free_energy`,
`belief_update`, and `precision_weight` for the same storage dtypes,
using fp32 scratch reductions for scalar free-energy objectives and
belief normalization. Metal Hawkes and Markov-blanket kernels cover
`hawkes_intensity`, `hawkes_kernel_matrix`,
`hawkes_log_likelihood`, `markov_mutual_information`,
`markov_blanket_partition`, `markov_flow_active`, and
`markov_flow_internal` for the same storage dtypes, with fp32 scratch
reductions for scalar objectives and dtype-native writes for vector
outputs. These families run through the device command queue with
async completion, pooled `MTLBuffer` storage, and per-kernel pipeline
caching.

Every backend implements the same interface, and the optimizer does the same math everywhere:

```go
type Runner interface {
    Execute(
        ctx context.Context,
        graph *ir.Graph,
        targets []*ir.Node,
    ) (map[string]tensor.Tensor, error)

    Location() tensor.Location
    Close() error
}
```

Backend kernels upload values once into a resident tensor store and only download at real boundaries. The executor releases owned dependencies after their last graph consumer; the host arena reuses released spans. Host-staged dispatch is restricted to the host backend, so Metal, CUDA, and XLA paths cannot silently route through CPU slices.

→ [Compute Backends](./docs/compute.md)

## 💾 Repository layout

```
cmd/                Cobra CLI: serve, chat, image, research
  asset/config.yml  The single config source
pkg/
  manifest/         YAML → IR compiler, registry, lowering
  runtime/          Manifest runtime programs, state, ops, schedulers, graph bridge
  backend/
    compute/        Runner interface + cpu/, cuda/, metal/, xla/
    api/            HTTP server
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

## 🔬 Testing

Every code file has a `_test.go` mirror. Tests are GoConvey-style ("Given X, it should Y", nested). Backend kernels run parity tests against the scalar reference at `N ∈ {1, 7, 64, 1024, 8192}` with tight ULP bounds — the tolerance is a contract, not a knob.

```bash
go test ./...
CGO_ENABLED=1 go test -tags cgo            ./pkg/backend/device/metal/...
CGO_ENABLED=1 go test -tags "cgo cuda"     ./pkg/backend/device/cuda/...
              go test -tags "cgo xla"      ./pkg/backend/device/xla/...
```

---

## 📓 Documentation

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
