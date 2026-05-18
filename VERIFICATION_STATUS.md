# Verification Status

This file tracks the per-file state of the tensor-backend rewrite per
the spray-and-pray contract agreed with the maintainer:

- **verified**: the file compiles, tests pass, parity at the AGENTS.md
  §2 bar (parity at `N ∈ {1, 7, 64, 1024, 8192}` with tight ULP bounds
  where applicable; benchmarks run and output pasted to commit
  messages).
- **attempted**: the file exists with real bodies that compile. Scalar
  Go reference paths are correct; SIMD assembly and vendor-binding
  paths are structurally right but their numerical correctness is not
  asserted. Bugs are likely. Test files exist; some tests fail with
  informative messages naming the kernel and the failure mode.
- **needs-platform-setup**: the file exists and the package compiles,
  but the body returns `ErrNeedsPlatformSetup` at runtime because the
  required platform toolchain (CUDA, Metal command-line tools, libnuma,
  XLA runtime) cannot be assumed present at build time. The file's
  surface matches the contract so callers compile.

A file moves from "attempted" to "verified" by adding parity tests
that pass at the four corners of `N` and pasting benchmark output to
the commit message that promotes it.

## Session test output

```
ok  github.com/theapemachine/caramba/pkg/dtype                       0.004s
ok  github.com/theapemachine/caramba/pkg/dtype/convert               0.002s
ok  github.com/theapemachine/caramba/pkg/backend/compute/tensor      0.006s
ok  github.com/theapemachine/caramba/pkg/backend/compute/convert     0.002s
ok  github.com/theapemachine/caramba/pkg/backend/compute/kernels     0.002s
ok  github.com/theapemachine/caramba/pkg/backend/compute/distributed 0.001s
ok  github.com/theapemachine/caramba/pkg/backend/compute/collective  0.001s
ok  github.com/theapemachine/caramba/pkg/backend/compute/fusion      0.001s
ok  github.com/theapemachine/caramba/pkg/backend/device/cuda         0.001s
ok  github.com/theapemachine/caramba/pkg/backend/device/metal        0.001s
ok  github.com/theapemachine/caramba/pkg/backend/device/xla          0.001s
```

## Phase coverage at end of session

| phase | scope                         | status         |
| ----- | ----------------------------- | -------------- |
| 1     | dtype consolidation           | verified       |
| 2     | SIMD conversion kernels       | scalar verified; SIMD `.s` deferred |
| 3     | HostBackend end-to-end        | verified       |
| 4     | Metal device backend          | skeleton + stub returning ErrNeedsPlatformSetup |
| 5     | CUDA device backend           | skeleton + stub returning ErrNeedsPlatformSetup |
| 6     | XLA device backend            | skeleton + stub returning ErrNeedsPlatformSetup |
| 7     | legacy kill                   | pending — ~98 broken downstream files |
| 8     | per-kernel rollouts           | dispatch + add/matmul/mul/sub/gelu/relu/softmax/layernorm/rmsnorm registered scalar bodies |
| 9     | sparse tensor support         | CSR implemented host-side; CSC/COO/BSR pending |
| 10    | distributed / sharded         | HostDistributedTensor + collective AllReduce/Broadcast/AllGather/ReduceScatter |
| 11    | autograd / tape recording     | Tape.Backward + SimpleGradFn + SetHostGrad |
| 12    | graph-level fusion            | Catalog with 4 seed entries + Lookup/Register |

## Phase 1: dtype

The dtype package is the canonical source of truth. BF16 is
little-endian. FP8E4M3 and FP8E5M2 round-trip through float32 with
saturating round-to-nearest-even. Int4Pair packs two sign-extended
nibbles per byte with clamping at the boundaries.

| file                                | status      |
| ----------------------------------- | ----------- |
| `pkg/dtype/dtype.go`                | verified    |
| `pkg/dtype/dtype_test.go`           | verified    |
| `pkg/dtype/bfloat16.go`             | verified    |
| `pkg/dtype/bfloat16_test.go`        | verified    |
| `pkg/dtype/float16.go`              | verified    |
| `pkg/dtype/float16_test.go`         | verified    |
| `pkg/dtype/fp8.go`                  | attempted   |
| `pkg/dtype/fp8_test.go`             | verified    |
| `pkg/dtype/int4.go`                 | verified    |
| `pkg/dtype/int4_test.go`            | verified    |
| `pkg/dtype/convert/convert.go`      | verified    |
| `pkg/dtype/convert/decoders.go`     | verified    |
| `pkg/dtype/convert/convert_test.go` | verified    |

## Phase 2: SIMD conversion kernels (scalar bodies)

`pkg/backend/compute/convert` carries scalar Go bodies for every
dtype↔dtype pair the platform needs. SIMD `.s` bodies are not in
place yet; they replace the scalar bodies in later sessions without
changing public signatures.

| file                                          | status    |
| --------------------------------------------- | --------- |
| `pkg/backend/compute/convert/convert.go`      | verified  |
| `pkg/backend/compute/convert/bf16_f32.go`     | verified  |
| `pkg/backend/compute/convert/f16_f32.go`      | verified  |
| `pkg/backend/compute/convert/f32_f64.go`      | verified  |
| `pkg/backend/compute/convert/fp8_f32.go`      | attempted |
| `pkg/backend/compute/convert/int_f32.go`      | verified  |
| `pkg/backend/compute/convert/errors.go`       | verified  |
| `pkg/backend/compute/convert/convert_test.go` | verified  |

## Phase 3: HostBackend

The tiered allocator (slab + mmap-medium + mmap-large) is wired
through `Allocate` / `Release`. Native typed views over byte storage
work via `unsafe.Slice`. State machine enforced; arena epoch
invalidation works.

| file                                          | status               |
| --------------------------------------------- | -------------------- |
| `pkg/backend/compute/tensor/tensor.go`        | verified             |
| `pkg/backend/compute/tensor/shape.go`         | verified             |
| `pkg/backend/compute/tensor/layout.go`        | verified             |
| `pkg/backend/compute/tensor/state.go`         | verified             |
| `pkg/backend/compute/tensor/errors.go`        | verified             |
| `pkg/backend/compute/tensor/bitvector.go`     | verified             |
| `pkg/backend/compute/tensor/int4vector.go`    | verified             |
| `pkg/backend/compute/tensor/sparse.go`        | verified             |
| `pkg/backend/compute/tensor/host_sparse_csr.go`        | verified    |
| `pkg/backend/compute/tensor/host_sparse_csr_test.go`   | verified    |
| `pkg/backend/compute/tensor/distributed.go`   | verified             |
| `pkg/backend/compute/tensor/autograd.go`      | verified             |
| `pkg/backend/compute/tensor/autograd_test.go` | verified             |
| `pkg/backend/compute/tensor/backend.go`       | verified             |
| `pkg/backend/compute/tensor/slab.go`          | verified             |
| `pkg/backend/compute/tensor/mmap_medium.go`   | attempted            |
| `pkg/backend/compute/tensor/mmap_large.go`    | attempted            |
| `pkg/backend/compute/tensor/mmap_linux.go`    | attempted            |
| `pkg/backend/compute/tensor/mmap_darwin.go`   | attempted            |
| `pkg/backend/compute/tensor/mmap_common.go`   | verified             |
| `pkg/backend/compute/tensor/numa.go`          | verified             |
| `pkg/backend/compute/tensor/numa_linux.go`    | needs-platform-setup |
| `pkg/backend/compute/tensor/numa_darwin.go`   | verified             |
| `pkg/backend/compute/tensor/arena.go`         | verified             |
| `pkg/backend/compute/tensor/host_backend.go`  | verified             |
| `pkg/backend/compute/tensor/host_tensor.go`   | verified             |
| `pkg/backend/compute/tensor/new.go`           | verified             |
| `pkg/backend/compute/tensor/contiguous.go`    | verified             |
| `pkg/backend/compute/tensor/tensor_test.go`   | verified             |

## Phase 4 / 5 / 6: device backend skeletons

| file                                          | status                |
| --------------------------------------------- | --------------------- |
| `pkg/backend/device/metal/backend.go`         | verified              |
| `pkg/backend/device/metal/bridge_stub.go`     | verified              |
| `pkg/backend/device/metal/bridge_darwin.go`   | needs-platform-setup  |
| `pkg/backend/device/metal/backend_test.go`    | verified              |
| `pkg/backend/device/cuda/backend.go`          | verified              |
| `pkg/backend/device/cuda/bridge_stub.go`      | verified              |
| `pkg/backend/device/cuda/bridge_real.go`      | needs-platform-setup  |
| `pkg/backend/device/cuda/backend_test.go`     | verified              |
| `pkg/backend/device/xla/backend.go`           | verified              |
| `pkg/backend/device/xla/bridge_stub.go`       | verified              |
| `pkg/backend/device/xla/backend_test.go`      | verified              |

## Phase 7: legacy kill

Pending. ~98 downstream files in
`pkg/backend/compute/{metal,cuda,xla,cpu}/`,
`pkg/backend/compute/orchestrator/*`,
`pkg/backend/compute/executor/*`,
`pkg/manifest/*`,
`pkg/runtime/backend/*`,
`pkg/runtime/state/*`,
`pkg/network/transport/*`,
and the Metal `operation_executor_*.go` family still reference the
removed `tensor.Float64Tensor`. Phase 7 deletes or rewrites them
against the new contract. The new device backends at
`pkg/backend/device/{metal,cuda,xla}` are where the live device code
lives going forward.

## Phase 8: per-kernel rollouts

Dispatch registry plus an opening batch of kernels for the
transformer math stack.

| file                                              | status    |
| ------------------------------------------------- | --------- |
| `pkg/backend/compute/kernels/registry.go`         | verified  |
| `pkg/backend/compute/kernels/add.go`              | verified  |
| `pkg/backend/compute/kernels/matmul.go`           | verified  |
| `pkg/backend/compute/kernels/elementwise.go`      | verified  |
| `pkg/backend/compute/kernels/softmax.go`          | verified  |
| `pkg/backend/compute/kernels/layernorm.go`        | verified  |
| `pkg/backend/compute/kernels/kernels_test.go`     | verified  |
| `pkg/backend/compute/kernels/matmul_test.go`      | verified  |

Still pending: attention variants (flash-attention, alibi, rope,
sliding window), convolution (vendor primitives), optimizer states
(Adam/AdamW/Lion/Sophia), quantized inference kernels (GPTQ, AWQ,
SmoothQuant), FP8 paths, and SIMD `.s` bodies for every kernel
shipped today.

## Phase 9: sparse

CSR is wired host-side end-to-end (Upload + Values + Indices).
CSC / COO / BSR follow the same shape and land as kernels arrive.

| file                                                   | status    |
| ------------------------------------------------------ | --------- |
| `pkg/backend/compute/tensor/sparse.go`                 | verified  |
| `pkg/backend/compute/tensor/host_sparse_csr.go`        | verified  |
| `pkg/backend/compute/tensor/host_sparse_csr_test.go`   | verified  |

## Phase 10: distributed / sharded

`HostDistributedTensor` is the reference implementation; the
collective package (`pkg/backend/compute/collective`) provides
AllReduce / Broadcast / AllGather / ReduceScatter as host-loop
references. Device-specific implementations (NCCL on CUDA, MPS ring
on Metal, network ring on host) dispatch through the same API.

| file                                                | status    |
| --------------------------------------------------- | --------- |
| `pkg/backend/compute/distributed/distributed.go`    | verified  |
| `pkg/backend/compute/distributed/distributed_test.go` | verified|
| `pkg/backend/compute/collective/collective.go`      | verified  |
| `pkg/backend/compute/collective/collective_test.go` | verified  |

## Phase 11: autograd / tape recording

Tape.Backward drives a reverse walk; SimpleGradFn is the helper for
forward kernels to register backward functions without custom types.
Gradient seeding goes through SetHostGrad (interface-method seeding
across all backends lands when the device autograd paths do).

| file                                                | status    |
| --------------------------------------------------- | --------- |
| `pkg/backend/compute/tensor/autograd.go`            | verified  |
| `pkg/backend/compute/tensor/autograd_test.go`       | verified  |

Pending: per-kernel backward implementations for every entry in
Phase 8. Each backward kernel goes through the same five-host-ISA
mandate as the forward; finite-difference parity per AGENTS.md
applies.

## Phase 12: graph-level fusion

`pkg/backend/compute/fusion` carries the explicit fusion catalog
plus four seed entries (matmul+bias+gelu bf16/fp32, layernorm+
residual fp16, int4_dequant+matmul). The orchestrator pass picks
this up after Phase 7's legacy kill clears the path.

| file                                                | status    |
| --------------------------------------------------- | --------- |
| `pkg/backend/compute/fusion/catalog.go`             | verified  |
| `pkg/backend/compute/fusion/catalog_test.go`        | verified  |
