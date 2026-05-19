# Backend coverage matrix (T1.5)

Combined snapshot of **T1.2** (`device.Backend` inventory), **T1.3** (CPU per-domain ISA dispatch), and **T1.4** (Metal / CUDA / XLA kernel registration). Counts are **registration** status from filesystem and `kernels.Default`; they do not assert full R1 implementation, parity, or legality on every required operation.

Machine-checkable source: `pkg/backend/device/coverageaudit/`, validated by `coverageaudit_test.go`.

## Detail documents

| Task | Document | Package |
|------|----------|----------|
| T1.2 | [`backend-inventory.md`](./backend-inventory.md) | `pkg/backend/device/inventory*.go` |
| T1.3 | [`cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md) | `pkg/backend/device/cpu/dispatchaudit/` |
| T1.4 | [`device-backend-matrix.md`](./device-backend-matrix.md) | `pkg/backend/device/backendaudit/` |

## R1 execution targets (registration snapshot)

Equal standing per `AGENTS.md` §1. **Applicable** is the audit denominator for that row; **registered** is what the combined audit currently finds.

| Target | Scope | Registered | Applicable | Notes |
|--------|-------|----------:|-----------:|-------|
| Go scalar | CPU domains | 30 | 30 | Go reference in every operation domain |
| AVX-512 | CPU domains (amd64) | 2 | 30 | Per-domain assembly/dispatch registration |
| AVX2 | CPU domains (amd64) | 2 | 30 | Per-domain assembly/dispatch registration |
| SSE2 | CPU domains (amd64) | 2 | 30 | Per-domain assembly/dispatch registration |
| NEON | CPU domains (arm64) | 20 | 30 | Per-domain assembly/dispatch registration |
| Metal | Required IR operations | 68 | 119 | 462 `kernels.Default` registrations total |
| CUDA | Required IR operations | 0 | 119 | Tensor upload/download; 0 kernel registrations |
| XLA | Required IR operations | 0 | 119 | Tensor upload/download; 0 kernel registrations |

## Backend inventory (T1.2)

| Item | Count |
|------|------:|
| `device.Backend` methods | 151 |
| `ir.RequiredOperationIDs()` | 119 |
| Required ops → direct `Backend` method | 74 |
| Required ops → composite `Backend` methods | 3 |
| Required ops → kernel registry | 32 |
| Required ops → graph-only | 10 |
| `Backend` methods with no required-op ID | 88 |

## CPU dispatch (T1.3)

| ISA path | Domains registered |
|----------|-------------------:|
| Scalar (Go) | 30 / 30 |
| AVX-512 (amd64) | 6 / 30 |
| AVX2 (amd64) | 2 / 30 |
| SSE2 (amd64) | 2 / 30 |
| NEON (arm64) | 20 / 30 |

AMD64 SIMD registered only on: `activation`, `pospop`.

Per-domain table: [`cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md).

## Device backends (T1.4)

| Backend | Kernel registrations | Required ops with Metal kernel |
|---------|------------------------:|-------------------------------:|
| Metal | 462 (158 unique names) | 68 / 119 |
| CUDA | 0 | — |
| XLA | 0 | — |

Per-kernel and per-operation tables: [`device-backend-matrix.md`](./device-backend-matrix.md).
