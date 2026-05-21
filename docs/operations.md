# Operations

Operations are the atomic units of computation in Caramba. Each operation has:

- A **YAML schema** in `pkg/asset/template/operation/` (defines inputs, outputs, config)
- A **CPU implementation** under `pkg/backend/device/cpu/` (Go + SIMD)
- A **CUDA implementation** under `pkg/backend/device/cuda/` (gated by `cgo cuda`)
- A **Metal implementation** under `pkg/backend/device/metal/` (gated by `darwin && cgo`)
- An **XLA implementation** under `pkg/backend/device/xla/` (gated by `cgo xla`)

No backend falls back silently. If a kernel is not implemented for a target backend, the build fails.

---

## Device backend contract

Every compute kernel ultimately implements a method on `device.Backend` in
[`puter/device/interface.go`](../../puter/device/interface.go). `Backend` embeds 25
interfaces (151 methods total). Manifest operation IDs map to these methods
directly, as composites, or via per-backend kernel registries (shape, optimizers).
See [`backend-inventory.md`](./backend-inventory.md) for the audited cross-link.

| Embedded interface | Methods |
|--------------------|--------:|
| `Activation`       |      55 |
| `Physics`          |       9 |
| `Causal`           |      10 |
| `Elementwise`      |      11 |
| `PosPop`           |       5 |
| `VSA`              |       5 |
| `Losses`           |       6 |
| `Hawkes`           |       5 |
| `Pool`             |       4 |
| `Convolution`      |       4 |
| `ActiveInference`  |       4 |
| `PredictiveCoding` |       4 |
| `Masking`          |       3 |
| `Attention`        |       3 |
| `Sampling`         |       3 |
| `Normalization`    |       3 |
| `Reduction`        |       5 |
| `LayerNorm`        |       2 |
| `RoPE`             |       2 |
| `Embedding`        |       2 |
| `Dequant`          |       2 |
| `Dot`              |       1 |
| `Matmul`           |       1 |
| `Dropout`          |       1 |
| `Quant`            |       1 |

In tables below, **manifest op** is the Caramba `op:` ID when one exists; **—** means
the method is on `device.Backend` only (no manifest template yet).

---

### PosPop

Population-count helpers over strings and fixed-width integer buffers.

| Backend method | Manifest op |
|----------------|-------------|
| `CountString`  | —           |
| `Count8`       | —           |
| `Count16`      | —           |
| `Count32`      | —           |
| `Count64`      | —           |

---

### Activation

Unary and gated activations; packed GLU variants take `packed` with `batch` × `halfCount`,
tensor GLU variants take separate `gate` and `up` buffers.

| Backend method     | Manifest op             |
|--------------------|-------------------------|
| `Exp`              | `math.exp`              |
| `Log`              | `math.log`              |
| `Log1p`            | —                       |
| `Expm1`            | —                       |
| `Sigmoid`          | `activation.sigmoid`    |
| `LogSigmoid`       | —                       |
| `Tanh`             | `activation.tanh`       |
| `Silu`             | —                       |
| `Swish`            | `activation.swish`      |
| `GeluTanh`         | —                       |
| `Gelu`             | `activation.gelu`       |
| `ReLU`             | `activation.relu`       |
| `LeakyReLU`        | `activation.leaky_relu` |
| `ELU`              | —                       |
| `CELU`             | —                       |
| `SELU`             | `activation.selu`       |
| `Softplus`         | —                       |
| `Mish`             | —                       |
| `Softsign`         | —                       |
| `HardSigmoid`      | —                       |
| `HardSwish`        | —                       |
| `HardTanh`         | —                       |
| `HardGelu`         | —                       |
| `QuickGelu`        | —                       |
| `TanhShrink`       | —                       |
| `Softmax`          | `math.softmax`          |
| `LogSoftmax`       | —                       |
| `PReLU`            | —                       |
| `PReLUV`           | —                       |
| `LeakyReLUSlope`   | —                       |
| `ELUAlpha`         | —                       |
| `CELUAlpha`        | —                       |
| `Threshold`        | —                       |
| `HardTanhRange`    | —                       |
| `Snake`            | —                       |
| `SnakeParametric`  | —                       |
| `HardShrink`       | —                       |
| `SoftShrink`       | —                       |
| `RReLU`            | —                       |
| `GLU`              | —                       |
| `GeGLU`            | —                       |
| `GeGLUTanh`        | —                       |
| `SwiGLU`           | `activation.swiglu`     |
| `ReGLU`            | —                       |
| `SiGLU`            | —                       |
| `GLUTensors`       | —                       |
| `GeGLUTensors`     | —                       |
| `GeGLUTanhTensors` | —                       |
| `SwiGLUTensors`    | —                       |
| `ReGLUTensors`     | —                       |
| `SiGLUTensors`     | —                       |
| `LinGLU`           | —                       |
| `SeGLU`            | —                       |
| `LinGLUTensors`    | —                       |
| `SeGLUTensors`     | —                       |

`Elementwise.ReLU` duplicates `Activation.ReLU` on the same `Backend` surface.

---

### Elementwise

| Backend method | Manifest op       |
|----------------|-------------------|
| `Add`          | `math.add`        |
| `Sub`          | —                 |
| `Mul`          | `math.mul`        |
| `Div`          | —                 |
| `Max`          | —                 |
| `Min`          | —                 |
| `Abs`          | —                 |
| `Neg`          | —                 |
| `Sqrt`         | —                 |
| `ReLU`         | `activation.relu` |
| `Axpy`         | —                 |

Additional elementwise helpers (`math.sin`, `math.cos`, `math.sign`, `math.outer`)
live in kernel registries and are not separate `Backend` methods.

---

### Reduction

| Backend method | Manifest op |
|----------------|-------------|
| `Sum`          | —           |
| `Prod`         | —           |
| `ReduceMin`    | —           |
| `ReduceMax`    | —           |
| `L1Norm`       | —           |

---

### Dot

| Backend method | Manifest op |
|----------------|-------------|
| `Dot`          | —           |

---

### Matmul

| Backend method | Manifest op   |
|----------------|---------------|
| `Matmul`       | `math.matmul` |

`projection.fused_qkv` composes `Matmul` with attention wiring (see
[`backend-inventory.md`](./backend-inventory.md)).

---

### Pool

| Backend method      | Manifest op                   |
|---------------------|-------------------------------|
| `MaxPool2D`         | `pooling.max_pool2d`          |
| `AvgPool2D`         | `pooling.avg_pool2d`          |
| `AdaptiveMaxPool2D` | `pooling.adaptive_max_pool2d` |
| `AdaptiveAvgPool2D` | `pooling.adaptive_avg_pool2d` |

---

### Convolution

| Backend method    | Manifest op                    |
|-------------------|--------------------------------|
| `Conv2D`          | `convolution.conv2d`           |
| `Conv1D`          | `convolution.conv1d`           |
| `Conv3D`          | `convolution.conv3d`           |
| `ConvTranspose2D` | `convolution.conv_transpose2d` |

---

### Dropout

| Backend method | Manifest op    |
|----------------|----------------|
| `Dropout`      | `math.dropout` |

---

### Losses

| Backend method       | Manifest op                |
|----------------------|----------------------------|
| `MSE`                | `train.loss.mse`           |
| `MAE`                | —                          |
| `Huber`              | —                          |
| `BinaryCrossEntropy` | —                          |
| `KLDivergence`       | —                          |
| `CrossEntropy`       | `train.loss.cross_entropy` |

Gradient variants (`train.grad.mse`, `train.grad.cross_entropy`) are kernel-registry
ops, not `Backend` methods.

---

### Sampling

| Backend method | Manifest op |
|----------------|-------------|
| `GreedySample` | —           |
| `TopKSample`   | —           |
| `TopPSample`   | —           |

---

### Embedding

| Backend method | Manifest op       |
|----------------|-------------------|
| `Lookup`       | `embedding.token` |
| `Bag`          | —                 |

---

### Normalization

| Backend method  | Manifest op |
|-----------------|-------------|
| `GroupNorm`     | —           |
| `InstanceNorm`  | —           |
| `BatchNormEval` | —           |

---

### LayerNorm

| Backend method | Manifest op      |
|----------------|------------------|
| `LayerNorm`    | `math.layernorm` |
| `RMSNorm`      | `math.rmsnorm`   |

---

### RoPE

| Backend method | Manifest op       |
|----------------|-------------------|
| `RoPE`         | `positional.rope` |
| `RoPEPairs`    | —                 |

---

### Hawkes

| Backend method            | Manifest op                         |
|---------------------------|-------------------------------------|
| `HawkesIntensity`         | `hawkes.intensity`                  |
| `HawkesKernelMatrix`      | `hawkes.kernel_matrix`              |
| `HawkesLogLikelihood`     | `hawkes.log_likelihood`             |
| `MarkovMutualInformation` | `markov_blanket.mutual_information` |
| `MarkovBlanketPartition`  | `markov_blanket.partition`          |

`hawkes.simulate` is a kernel-registry op (not on `Backend`).

---

### Physics

Spatial-stencil and quantum-hydrodynamics operators. Unlike thermodynamic blocks,
these are real `Backend` kernels (no `shape.shift` / `shape.roll` primitive exists).

| Backend method       | Manifest op         |
|----------------------|---------------------|
| `Laplacian`          | `stencil.laplacian` |
| `Laplacian4`         | —                   |
| `Grad1D`             | —                   |
| `Divergence1D`       | —                   |
| `FFT1D`              | —                   |
| `IFFT1D`             | —                   |
| `QuantumPotential`   | —                   |
| `BohmianVelocity`    | —                   |
| `MadelungContinuity` | —                   |

---

### Causal

| Backend method           | Manifest op                       |
|--------------------------|-----------------------------------|
| `Cholesky`               | —                                 |
| `BackdoorAdjustment`     | `causal.backdoor_adjustment`      |
| `FrontdoorAdjustment`    | `causal.frontdoor_adjustment`     |
| `DoIntervene`            | `causal.do_calculus`              |
| `CATE`                   | `causal.cate`                     |
| `Counterfactual`         | `causal.counterfactual`           |
| `IVEstimate`             | `causal.iv_estimate`              |
| `DAGMarkovFactorization` | `causal.dag_markov_factorization` |
| `MarkovFlowActive`       | `markov_blanket.flow_active`      |
| `MarkovFlowInternal`     | `markov_blanket.flow_internal`    |

---

### Masking

| Backend method | Manifest op        |
|----------------|--------------------|
| `ApplyMask`    | `masking.apply`    |
| `CausalMask`   | `masking.causal`   |
| `ALiBiBias`    | `positional.alibi` |

---

### Attention

| Backend method              | Manifest op                                  |
|-----------------------------|----------------------------------------------|
| `ScaledDotProductAttention` | `attention.sdpa`                             |
| `FlashAttention`            | —                                            |
| `MultiHeadAttention`        | `attention.mqa`, `attention.gqa` (composite) |

`attention.sliding_window` is implemented via attention config / graph wiring, not a
separate `Backend` method.

---

### VSA

| Backend method   | Manifest op           |
|------------------|-----------------------|
| `Bind`           | `vsa.bind`            |
| `Bundle`         | `vsa.bundle`          |
| `Permute`        | `vsa.permute`         |
| `InversePermute` | `vsa.inverse_permute` |
| `Similarity`     | `vsa.similarity`      |

---

### ActiveInference

| Backend method       | Manifest op                             |
|----------------------|-----------------------------------------|
| `FreeEnergy`         | `active_inference.free_energy`          |
| `ExpectedFreeEnergy` | `active_inference.expected_free_energy` |
| `BeliefUpdate`       | `active_inference.belief_update`        |
| `PrecisionWeight`    | `active_inference.precision_weight`     |

---

### PredictiveCoding

| Backend method         | Manifest op                               |
|------------------------|-------------------------------------------|
| `Prediction`           | `predictive_coding.prediction`            |
| `PredictionError`      | `predictive_coding.prediction_error`      |
| `UpdateRepresentation` | `predictive_coding.update_representation` |
| `UpdateWeights`        | `predictive_coding.update_weights`        |

---

### Dequant

| Backend method | Manifest op |
|----------------|-------------|
| `Dequant`      | —           |
| `Dequant4`     | —           |

---

### Quant

| Backend method | Manifest op |
|----------------|-------------|
| `Quant`        | —           |

---

## Manifest-only operations

These appear in manifests and templates but are **not** methods on `device.Backend`
(optimizers, shape, control, tokenizer, model editing, benchmarks, etc.).

### Attention (graph / config)

| Op ID                      | Description                    |
|----------------------------|--------------------------------|
| `attention.sdpa`           | Scaled dot-product attention   |
| `attention.gqa`            | Grouped-query attention        |
| `attention.mqa`            | Multi-query attention          |
| `attention.sliding_window` | Sliding-window local attention |

### Tokenizer

| Op ID              | Description                     |
|--------------------|---------------------------------|
| `tokenizer.load`   | Load a tokenizer.json artifact  |
| `tokenizer.encode` | Encode prompt text to token IDs |
| `tokenizer.decode` | Decode token IDs to text        |

### Projection

| Op ID                  | Description                                |
|------------------------|--------------------------------------------|
| `projection.linear`    | Linear projection (weight + optional bias) |
| `projection.fused_qkv` | Fused Q/K/V projection                     |
| `projection.moe`       | Mixture-of-experts routing                 |
| `model.lora`           | Low-rank adaptation (LoRA)                 |

### Shape & control

| Op ID                 | Description                     |
|-----------------------|---------------------------------|
| `shape.reshape`       | Tensor reshape                  |
| `shape.transpose`     | Tensor transpose                |
| `shape.concat`        | Concatenate along a dimension   |
| `shape.split`         | Split along a dimension         |
| `shape.slice`         | Contiguous range extraction     |
| `shape.view_as_heads` | Reshape for multi-head layout   |
| `shape.merge_heads`   | Merge multi-head layout         |
| `control.repeat`      | Repeat tensor along a dimension |

Metal `shape.slice` is handled by `operation_executor.applySlice`, which currently
requires `start==0` and leading-dim slicing. The Metal failure maps to:

```text
metal tensor: slice node %q currently supports start=0 with leading-dim slicing only (got start=%d, dim=%d, outer=%d)
```

Supported on Metal:

```yaml
op: shape.slice
config: { dim: 1, start: 0, end: 4096 } # shape [1, 4112, 64], outer=1
```

Unsupported on Metal:

```yaml
op: shape.slice
config: { dim: 1, start: 64, end: 128 } # start != 0
```

Run non-prefix slices on CPU or avoid them until Metal strided-copy support exists.

### Math helpers (kernel registry)

| Op ID                     | Description           |
|---------------------------|-----------------------|
| `math.sin`                | Elementwise sine      |
| `math.cos`                | Elementwise cosine    |
| `math.sign`               | Elementwise sign      |
| `math.outer`              | Outer product         |
| `math.logsumexp`          | Log-sum-exp reduction |
| `math.inv_sqrt_dim_scale` | `1/sqrt(dim)` scaling |

### Model & data

| Op ID                   | Description                  |
|-------------------------|------------------------------|
| `model.load`            | Load weights checkpoint      |
| `model.freeze`          | Freeze parameter nodes       |
| `model.graft`           | Graft subgraph weights       |
| `model.adapter`         | Adapter injection            |
| `model.surgery`         | Structural model surgery     |
| `data.huggingface`      | Hugging Face dataset binding |
| `train.checkpoint.load` | Training checkpoint load     |
| `train.checkpoint.save` | Training checkpoint save     |

### Benchmarks

| Op ID                     | Description               |
|---------------------------|---------------------------|
| `bench.metric.accuracy`   | Classification accuracy   |
| `bench.metric.perplexity` | Language-model perplexity |
| `bench.metric.f1`         | F1 score                  |

### Hawkes (registry)

| Op ID             | Description                     |
|-------------------|---------------------------------|
| `hawkes.simulate` | Simulate Hawkes event sequences |

---

## Research / esoteric blocks

Composite blocks in `pkg/asset/template/block/` bundle resident ops; they do not add
`device.Backend` methods.

### Energy-based models

| Block ID                              | Description                                |
|---------------------------------------|--------------------------------------------|
| `block.energy.boltzmann_distribution` | Energies → Boltzmann probabilities         |
| `block.energy.free_energy`            | `-beta^{-1} logsumexp(-beta E)`            |
| `block.energy.langevin_step`          | One externally differentiated sampler step |
| `block.energy.contrastive_phase`      | Per-sample positive/negative phase deltas  |

---

## Optimizers

Optimizer steps are required IR ops but live in per-backend kernel registries, not on
`device.Backend`. Templates: `pkg/asset/template/operation/train/optimizer/`.

| Optimizer | Op ID                      |
|-----------|----------------------------|
| Adam      | `train.optimizer.adam`     |
| AdamW     | `train.optimizer.adamw`    |
| SGD       | `train.optimizer.sgd`      |
| Lion      | `train.optimizer.lion`     |
| AdaGrad   | `train.optimizer.adagrad`  |
| AdaDelta  | `train.optimizer.adadelta` |
| AdaMax    | `train.optimizer.adamax`   |
| RMSProp   | `train.optimizer.rmsprop`  |
| LAMB      | `train.optimizer.lamb`     |
| LARS      | `train.optimizer.lars`     |
| L-BFGS    | `train.optimizer.lbfgs`    |
| Hebbian   | `train.optimizer.hebbian`  |

---

## Operation schema format

Each operation is described by a YAML schema that drives the frontend node graph editor
and the manifest compiler:

```yaml
# pkg/asset/template/operation/attention/sdpa.yml
kind: Operation
category: attention
op: attention.sdpa
name: Scaled Dot-Product Attention

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

## Adding a new operation

1. **Add the YAML schema** in `pkg/asset/template/operation/<category>/<name>.yml`
2. **Implement the CPU kernel** under `pkg/backend/device/cpu/`
   - Pure Go reference (always)
   - SIMD assembly where performance-critical (`_amd64.s`, `_arm64.s`)
3. **Wire `device.Backend`** — add or extend a method in
   [`puter/device/interface.go`](../../puter/device/interface.go) when the op is compute
4. **Implement CUDA** under `pkg/backend/device/cuda/`
5. **Implement Metal** under `pkg/backend/device/metal/`
6. **Implement XLA** under `pkg/backend/device/xla/`
7. **Register** the operation in the manifest registry
8. **Write tests** in a `_test.go` mirror, GoConvey, parity against scalar reference

The operation is then available in manifests as `<category>.<name>` and in the frontend
operation picker.

---

## Template blocks

Beyond individual operations, Caramba provides pre-wired **blocks**—composite
subgraphs that appear as a single collapsed node. Blocks live in
`pkg/asset/template/block/`:

| Block category      | Examples                                                |
|---------------------|---------------------------------------------------------|
| `active_inference`  | Free energy minimization block                          |
| `causal`            | Causal temporal block                                   |
| `energy`            | Boltzmann normalization, EBM free energy, sampler steps |
| `hawkes`            | Hawkes process attention block                          |
| `markov_blanket`    | Markov blanket hierarchy block                          |
| `memory`            | External memory read/write block                        |
| `predictive_coding` | Hierarchical prediction error block                     |
| `vsa`               | VSA bind-bundle-query block                             |

Full model templates live in `pkg/asset/template/model/` (`llm`, `vision`, `audio`,
`diffusion`, etc.).
