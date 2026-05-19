# Backend method inventory (T1.2)

Cross-reference between `device.Backend` embedded interface methods and
`ir.RequiredOperationIDs()`. Machine-checkable source of truth:
`pkg/backend/device/inventory.go`, `inventory_links.go`, validated by
`pkg/backend/device/inventory_test.go`.

CPU per-domain dispatch registration (T1.3): [`docs/cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md)
(`pkg/backend/device/cpu/dispatchaudit/`).

Metal / CUDA / XLA tensor backend matrix (T1.4): [`docs/device-backend-matrix.md`](./device-backend-matrix.md)
(`pkg/backend/device/backendaudit/`).

Combined ISA/backend coverage (T1.5): [`backend-coverage.md`](./backend-coverage.md)
(`pkg/backend/device/coverageaudit/`).

## Summary (2026-05-19)

| Item | Count |
|------|------:|
| `device.Backend` methods (25 embedded interfaces) | 151 |
| `ir.RequiredOperationIDs()` entries | 119 |
| Required ops → direct `Backend` method | 74 |
| Required ops → composite `Backend` methods | 3 |
| Required ops → kernel registry (not on `Backend`) | 32 |
| Required ops → graph-only (no compute kernel) | 10 |
| `Backend` methods with no required-op ID | 88 |

## Embedded surfaces

| Surface | Methods |
|---------|--------:|
| Activation | 55 |
| Physics | 9 |
| Causal | 10 |
| Elementwise | 11 |
| PosPop | 5 |
| VSA | 5 |
| Losses | 6 |
| Hawkes | 5 |
| Pool | 4 |
| Convolution | 4 |
| ActiveInference | 4 |
| PredictiveCoding | 4 |
| Masking | 3 |
| Attention | 3 |
| Sampling | 3 |
| Normalization | 3 |
| Reduction | 5 |
| LayerNorm | 2 |
| RoPE | 2 |
| Embedding | 2 |
| Dequant | 2 |
| Dot | 1 |
| Matmul | 1 |
| Dropout | 1 |
| Quant | 1 |

## Cross-link kinds

### Direct (74)

Each required operation ID maps to a single `Surface.Method` on `device.Backend`.
Examples: `math.add` → `Elementwise.Add`, `convolution.conv2d` → `Convolution.Conv2D`,
`train.loss.mse` → `Losses.MSE`.

### Composite (3)

| Operation ID | Backend methods |
|----------------|-----------------|
| `attention.mqa` | `Attention.MultiHeadAttention`, `Attention.FlashAttention` |
| `attention.gqa` | `Attention.MultiHeadAttention`, `Attention.FlashAttention` |
| `projection.fused_qkv` | `Matmul.Matmul`, `Attention.MultiHeadAttention` |

### Kernel registry (32)

Implemented under CPU/Metal/CUDA kernel registries, not as `device.Backend` methods:

- **Shape (9):** `shape.reshape`, `shape.transpose`, `shape.concat`, `shape.split`,
  `shape.upsample_nearest2d`, `shape.view_as_heads`, `shape.merge_heads`,
  `shape.last_token`, `shape.slice` → `pkg/backend/device/cpu/shape`
- **Optimizers (12):** `train.optimizer.*` → `pkg/backend/device/cpu/optimizer`,
  `pkg/backend/device/metal` optimizer kernels (not on `Backend`)
- **Training grads (4):** `train.loss.*_grad`, `train.grad.*` → losses kernel registry
- **Math helpers (4):** `math.sin`, `math.cos`, `math.outer`, `math.sign` → elementwise/matmul registries
- **Other (3):** `math.logsumexp`, `hawkes.simulate`, `math.inv_sqrt_dim_scale`

### Graph-only (10)

| Operation ID | Note |
|----------------|------|
| `OpInput` | Graph input binding |
| `OpFused` | Fusion pass composite |
| `bench.accuracy`, `bench.perplexity`, `bench.f1` | Metric nodes |
| `bench.metric.accuracy`, `bench.metric.perplexity`, `bench.metric.f1` | Metric nodes |
| `model.graft`, `model.freeze` | Model-editing graph ops |

## Backend-only methods (88)

Methods on `device.Backend` not referenced by any `ir.RequiredOperationIDs()` entry.
Includes extended activations (`Activation.Exp`, `Activation.GeluTanh`, gated tensor
variants), `PosPop`, `Physics`, `Sampling`, `Dequant`/`Quant`, duplicate
`Elementwise.ReLU`, and most of the `Activation` parametric variants.

Use `device.BackendMethodsWithoutRequiredOperation()` for the authoritative list.

## Gaps for R1

- Optimizer steps are required IR ops but are **not** on `device.Backend`; they live in
  per-backend kernel registries.
- Shape ops are required IR ops but are **not** on `device.Backend`.
- 88 `Backend` methods have no required-op ID (extended/specialized surfaces).
