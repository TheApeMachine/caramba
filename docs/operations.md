# Operations

Operations are the atomic units of computation in Caramba. Each operation has:

- A **YAML schema** in `pkg/asset/template/operation/` (defines inputs, outputs, config)
- A **CPU implementation** in `pkg/backend/compute/cpu/operation/` (Go + SIMD)
- A **CUDA implementation** in `pkg/backend/compute/cuda/` (gated by `cgo cuda`)
- A **Metal implementation** in `pkg/backend/compute/metal/` (gated by `darwin && cgo`)
- An **XLA implementation** in `pkg/backend/compute/xla/` (gated by `cgo xla`)

No backend falls back silently. If a kernel isn't implemented for a target backend, the build fails.

---

## Standard Deep Learning

### Activation

| Op ID                  | Description                          |
|------------------------|--------------------------------------|
| `activation.relu`      | Rectified linear unit                |
| `activation.leaky_relu`| Leaky ReLU with configurable slope   |
| `activation.gelu`      | Gaussian error linear unit           |
| `activation.selu`      | Scaled exponential linear unit       |
| `activation.sigmoid`   | Logistic sigmoid                     |
| `activation.swiglu`    | Swish-gated linear unit              |
| `activation.swish`     | Swish (SiLU)                         |
| `activation.tanh`      | Hyperbolic tangent                   |

### Attention

| Op ID                       | Description                                   |
|-----------------------------|-----------------------------------------------|
| `attention.sdpa`            | Scaled dot-product attention                  |
| `attention.gqa`             | Grouped-query attention                       |
| `attention.mqa`             | Multi-query attention                         |
| `attention.sliding_window`  | Sliding-window local attention                |

### Embedding

| Op ID                   | Description                                  |
|-------------------------|----------------------------------------------|
| `embedding.token`       | Learned token embeddings                     |
| `embedding.rope`        | Rotary position embeddings (RoPE)            |
| `embedding.alibi`       | Attention with linear biases (ALiBi)         |
| `embedding.sinusoidal`  | Fixed sinusoidal position encoding           |

### Normalization

| Op ID              | Description                    |
|--------------------|--------------------------------|
| `norm.layer`       | Layer normalization            |
| `norm.rms`         | RMS normalization              |
| `norm.batch`       | Batch normalization            |

### Projection

| Op ID                   | Description                                  |
|-------------------------|----------------------------------------------|
| `projection.linear`     | Linear projection (weight + optional bias)   |
| `projection.fused_qkv`  | Fused Q/K/V projection                       |
| `projection.lora`       | Low-rank adaptation (LoRA)                   |

### Convolution

| Op ID            | Description                   |
|------------------|-------------------------------|
| `conv.1d`        | 1D convolution                |
| `conv.2d`        | 2D convolution                |
| `conv.depthwise` | Depthwise separable conv      |
| `conv.grouped`   | Grouped convolution           |

### Pooling

| Op ID           | Description              |
|-----------------|--------------------------|
| `pool.mean`     | Mean pooling             |
| `pool.max`      | Max pooling              |
| `pool.attention`| Attention-weighted pooling |

### Masking & Shape

| Op ID            | Description                    |
|------------------|--------------------------------|
| `mask.causal`    | Causal (autoregressive) mask   |
| `mask.padding`   | Padding mask                   |
| `shape.reshape`  | Tensor reshape                 |
| `shape.transpose`| Tensor transpose               |
| `shape.expand`   | Tensor broadcast expansion     |

### Math

| Op ID         | Description              |
|---------------|--------------------------|
| `math.softmax`| Softmax normalization    |
| `math.add`    | Elementwise addition     |
| `math.mul`    | Elementwise multiplication |
| `math.matmul` | Matrix multiplication    |

---

## Research / Esoteric Architectures

### Active Inference

Implements free energy minimization and precision-weighted prediction error. Used for architectures grounded in the Free Energy Principle.

| Op ID                            | Description                           |
|----------------------------------|---------------------------------------|
| `active_inference.free_energy`   | Variational free energy computation   |
| `active_inference.precision`     | Precision-weighted prediction error   |
| `active_inference.belief_update` | Bayesian belief updating              |

### Causal Inference

| Op ID                      | Description                          |
|----------------------------|--------------------------------------|
| `causal.temporal_diff`     | Temporal difference learning         |
| `causal.intervention`      | Causal intervention (do-calculus)    |
| `causal.counterfactual`    | Counterfactual estimation            |

### Hawkes Process

Point process kernels for modeling event sequences with excitatory interactions.

| Op ID                   | Description                          |
|-------------------------|--------------------------------------|
| `hawkes.intensity`      | Conditional intensity computation    |
| `hawkes.attention`      | Hawkes process attention kernel      |
| `hawkes.log_likelihood` | Point process log-likelihood         |

### Markov Blanket

Free energy decomposition and blanket detection for hierarchical models grounded in active inference.

| Op ID                        | Description                          |
|------------------------------|--------------------------------------|
| `markov_blanket.partition`   | Blanket partition detection          |
| `markov_blanket.flow_internal` | Internal state flow               |
| `markov_blanket.flow_active` | Active state flow                    |
| `markov_blanket.mutual_info` | Mutual information between blankets  |

The Markov blanket kernels have dedicated ARM64 NEON assembly for performance-critical paths.

### Predictive Coding

Hierarchical prediction error propagation for architectures inspired by Rao & Ballard's predictive coding theory.

| Op ID                         | Description                          |
|-------------------------------|--------------------------------------|
| `predictive_coding.predict`   | Top-down prediction generation       |
| `predictive_coding.error`     | Prediction error computation         |
| `predictive_coding.update`    | Precision-weighted weight update     |

### Vector Symbolic Architecture (VSA)

Hyperdimensional computing operations for compositional symbolic-subsymbolic integration.

| Op ID             | Description                            |
|-------------------|----------------------------------------|
| `vsa.bind`        | Binding (XOR, circular convolution)    |
| `vsa.bundle`      | Bundling (superposition)               |
| `vsa.permute`     | Permutation (shifting role encoding)   |
| `vsa.inv_permute` | Inverse permutation                    |
| `vsa.similarity`  | Cosine similarity / cleanup memory     |

---

## Optimizers

Optimizers live in `pkg/backend/compute/cpu/optimizer/` and `pkg/asset/template/optimizer/`:

| Optimizer   | Notes                                              |
|-------------|-----------------------------------------------------|
| `adam`      | AdamW with decoupled weight decay                  |
| `sgd`       | Stochastic gradient descent with momentum          |
| `lion`      | Sign-based optimizer (memory-efficient)            |
| `adagrad`   | Adaptive gradient                                  |
| `rmsprop`   | RMSProp                                            |
| `lbfgs`     | Limited-memory BFGS (second-order)                 |
| `lars`      | Layer-wise adaptive rate scaling                   |
| `hebbian`   | Hebbian learning (unsupervised)                    |

---

## Operation Schema Format

Each operation is described by a YAML schema that drives the frontend node graph editor and the manifest compiler:

```yaml
# pkg/asset/template/operation/attention/sdpa.yml
id: attention.sdpa
name: Scaled Dot-Product Attention
category: attention

inputs:
  - name: query
    type: tensor
    required: true
  - name: key
    type: tensor
    required: true
  - name: value
    type: tensor
    required: true
  - name: mask
    type: mask
    required: false

outputs:
  - name: out
    type: tensor

config:
  d_model:
    type: int
    required: true
  n_heads:
    type: int
    required: true
  dropout:
    type: float
    default: 0.0
  causal:
    type: bool
    default: false
```

---

## Adding a New Operation

1. **Add the YAML schema** in `pkg/asset/template/operation/<category>/<name>.yml`
2. **Implement the CPU kernel** in `pkg/backend/compute/cpu/operation/<category>/`
   - Pure Go implementation (always)
   - SIMD assembly where performance-critical (`_amd64.s`, `_arm64.s`)
3. **Implement CUDA kernel** in `pkg/backend/compute/cuda/`
4. **Implement Metal kernel** in `pkg/backend/compute/metal/`
5. **Implement XLA kernel** in `pkg/backend/compute/xla/`
6. **Register** the operation in `pkg/manifest/registry.go`
7. **Write tests** in a `_test.go` file alongside the implementation, using Goconvey

The operation is now available in manifests as `<category>.<name>` and appears in the frontend node graph editor's operation picker.

---

## Template Blocks

Beyond individual operations, Caramba provides pre-wired **blocks**—composite subgraphs that appear as a single collapsed node in the editor. Blocks are defined in `pkg/asset/template/block/`:

| Block category     | Examples                                             |
|--------------------|------------------------------------------------------|
| `active_inference` | Free energy minimization block                       |
| `causal`           | Causal temporal block                                |
| `hawkes`           | Hawkes process attention block                       |
| `markov_blanket`   | Markov blanket hierarchy block                       |
| `memory`           | External memory read/write block                     |
| `predictive_coding`| Hierarchical prediction error block                  |
| `vsa`              | VSA bind-bundle-query block                          |

Full model templates (entire architectures as blocks) live in `pkg/asset/template/model/`:

| Model category | Examples                                          |
|----------------|---------------------------------------------------|
| `llm`          | Llama 3.2, GPT-2, Mistral, Phi                   |
| `vision`       | ViT, CLIP visual encoder                         |
| `audio`        | Whisper encoder                                   |
| `diffusion`    | UNet, DiT                                         |
