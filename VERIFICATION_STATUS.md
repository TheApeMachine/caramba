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
that pass at the five required `N` sizes and pasting benchmark output
to the commit message that promotes it.

## Session test output

### 2026-05-18 Metal device kernel slice

Focused Metal device tests:

```
=== RUN   TestNewBackend

  Given the Metal backend constructor ✔✔


2 total assertions

--- PASS: TestNewBackend (0.04s)
=== RUN   TestBackend_UploadDownloadFloat32

  Given a Metal float32 tensor upload ✔✔✔✔✔✔✔


9 total assertions

--- PASS: TestBackend_UploadDownloadFloat32 (0.00s)
=== RUN   TestBackend_AddFloat32
=== RUN   TestBackend_AddFloat32/N=1

  Given two Metal float32 tensors ✔✔✔✔✔✔✔✔


17 total assertions

=== RUN   TestBackend_AddFloat32/N=7

  Given two Metal float32 tensors ✔✔✔✔✔✔✔✔


25 total assertions

=== RUN   TestBackend_AddFloat32/N=64

  Given two Metal float32 tensors ✔✔✔✔✔✔✔✔


33 total assertions

=== RUN   TestBackend_AddFloat32/N=1024

  Given two Metal float32 tensors ✔✔✔✔✔✔✔✔


41 total assertions

=== RUN   TestBackend_AddFloat32/N=8192

  Given two Metal float32 tensors ✔✔✔✔✔✔✔✔


49 total assertions

--- PASS: TestBackend_AddFloat32 (0.01s)
    --- PASS: TestBackend_AddFloat32/N=1 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=7 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=64 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=1024 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=8192 (0.00s)
=== RUN   TestKernelRegistry_MetalAddFloat32

  Given the device kernel registry ✔✔✔✔✔✔✔✔✔✔


59 total assertions

--- PASS: TestKernelRegistry_MetalAddFloat32 (0.01s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.653s
```

Metal library generator tests:

```
=== RUN   TestNewGenerator

  Given a Metal library generator ✔✔


2 total assertions

--- PASS: TestNewGenerator (0.00s)
=== RUN   TestGenerator_MetalArgs

  Given a Metal library generator ✔


3 total assertions

--- PASS: TestGenerator_MetalArgs (0.00s)
=== RUN   TestGenerator_MetallibArgs

  Given a Metal library generator ✔


4 total assertions

--- PASS: TestGenerator_MetallibArgs (0.00s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.341s
```

Focused package sweep:

```
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.302s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.959s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.758s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	1.277s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	0.471s
```

Metal benchmark output:

```
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkNewBackend-16              	    8492	    142513 ns/op	    1264 B/op	       4 allocs/op
BenchmarkBackend_AddFloat32/N=1-16  	    9727	    116965 ns/op	   0.10 MB/s	    1521 B/op	       6 allocs/op
BenchmarkBackend_AddFloat32/N=7-16  	   10000	    111639 ns/op	   0.75 MB/s	    1520 B/op	       6 allocs/op
BenchmarkBackend_AddFloat32/N=64-16 	   10000	    111633 ns/op	   6.88 MB/s	    1520 B/op	       6 allocs/op
BenchmarkBackend_AddFloat32/N=1024-16         	   10000	    110222 ns/op	 111.48 MB/s	    1520 B/op	       6 allocs/op
BenchmarkBackend_AddFloat32/N=8192-16         	   10000	    106858 ns/op	 919.95 MB/s	    1520 B/op	       6 allocs/op
BenchmarkKernel_RunAddFloat32/N=1-16          	   10594	    113950 ns/op	   0.11 MB/s	    1264 B/op	       2 allocs/op
BenchmarkKernel_RunAddFloat32/N=7-16          	   10000	    114260 ns/op	   0.74 MB/s	    1264 B/op	       2 allocs/op
BenchmarkKernel_RunAddFloat32/N=64-16         	   10000	    114239 ns/op	   6.72 MB/s	    1264 B/op	       2 allocs/op
BenchmarkKernel_RunAddFloat32/N=1024-16       	   10000	    115498 ns/op	 106.39 MB/s	    1264 B/op	       2 allocs/op
BenchmarkKernel_RunAddFloat32/N=8192-16       	   10000	    111902 ns/op	 878.49 MB/s	    1264 B/op	       2 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	12.912s
```

This slice adds `pkg/backend/device/metal/add_float32.metal`,
`pkg/backend/device/metal/kernels.metallib`, a reproducible
`go generate ./pkg/backend/device/metal` path, a pooled `MTLBuffer`
upload/download path with finalizers, a `newLibraryWithData` Metal
library load, asynchronous command submission with completion callbacks
and tensor readiness tracking, `threadExecutionWidth` threadgroups, a
`float4` vectorized add body with scalar tail handling, and a
Metal-specific kernel registry entry resolved through `LookupLocation`.
Metal capabilities report `SupportsAsync: false` because upload returns
a ready tensor; compute dispatch itself is asynchronous.

### 2026-05-18 Phase 7 slice

```
ok   github.com/theapemachine/caramba/pkg/dtype
ok   github.com/theapemachine/caramba/pkg/dtype/convert
ok   github.com/theapemachine/caramba/pkg/backend/compute/tensor
ok   github.com/theapemachine/caramba/pkg/backend/compute/kernels
ok   github.com/theapemachine/caramba/pkg/backend/compute/ir
ok   github.com/theapemachine/caramba/pkg/backend/compute/state
?    github.com/theapemachine/caramba/pkg/backend/compute/runner [no test files]
ok   github.com/theapemachine/caramba/pkg/backend/compute/executor
ok   github.com/theapemachine/caramba/pkg/backend/compute/cpu
ok   github.com/theapemachine/caramba/pkg/backend/compute/dispatch
ok   github.com/theapemachine/caramba/pkg/backend/compute/orchestrator
ok   github.com/theapemachine/caramba/pkg/backend/compute
ok   github.com/theapemachine/caramba/pkg/network/transport
ok   github.com/theapemachine/caramba/pkg/model/weights
ok   github.com/theapemachine/caramba/pkg/runtime/state
ok   github.com/theapemachine/caramba/pkg/manifest
ok   github.com/theapemachine/caramba/pkg/runtime/backend
```

Focused benchmark output:

```
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/compute/executor
cpu: Apple M4 Max
BenchmarkExecutor_Execute-16        589704      2032 ns/op    5768 B/op    48 allocs/op
PASS
ok   github.com/theapemachine/caramba/pkg/backend/compute/executor 1.432s

goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/compute/cpu
cpu: Apple M4 Max
BenchmarkTensorBackend_ApplyMatmul-16 10846   109519 ns/op  240080 B/op   31 allocs/op
PASS
ok   github.com/theapemachine/caramba/pkg/backend/compute/cpu 1.546s

goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/compute/orchestrator
cpu: Apple M4 Max
BenchmarkScheduler/SimpleGraph-16   60033     19686 ns/op   33878 B/op   132 allocs/op
PASS
ok   github.com/theapemachine/caramba/pkg/backend/compute/orchestrator 4.181s

goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/compute
cpu: Apple M4 Max
BenchmarkBackend_Execute-16         57862     20970 ns/op   41754 B/op   194 allocs/op
PASS
ok   github.com/theapemachine/caramba/pkg/backend/compute 1.562s
```

Phase 7 moved these packages from the legacy float64 tensor API to
`tensor.Tensor` / `dtype.DType`: `pkg/backend/compute/ir`,
`pkg/backend/compute/state`, `pkg/backend/compute/runner`,
`pkg/backend/compute/executor`, `pkg/backend/compute/cpu`,
`pkg/backend/compute/dispatch`, `pkg/backend/compute/orchestrator`,
`pkg/backend/compute`, `pkg/network/transport`, `pkg/manifest`, and
`pkg/runtime/backend`.

Remaining legacy references:

```
rg -l "Float64Tensor|UploadFloat64|DownloadFloat64|CloneFloat64|Float64From|MustFloat64From|MustCloneFloat64|tensor\\.DType|tensor\\.Float64|tensor\\.Float32" pkg README.md docs | wc -l
113
```

The remaining references are not complete. The largest surfaces are
the old `pkg/backend/compute/{cuda,metal,xla}` packages, runtime
network/output paths outside the touched slice, and documentation.

### Previous session

```
ok  github.com/theapemachine/caramba/pkg/dtype                       0.004s
ok  github.com/theapemachine/caramba/pkg/dtype/convert               0.002s
ok  github.com/theapemachine/caramba/pkg/backend/compute/tensor      0.008s
ok  github.com/theapemachine/caramba/pkg/backend/compute/convert     0.066s
ok  github.com/theapemachine/caramba/pkg/backend/compute/kernels     0.045s
ok  github.com/theapemachine/caramba/pkg/backend/compute/distributed 0.003s
ok  github.com/theapemachine/caramba/pkg/backend/compute/collective  0.005s
ok  github.com/theapemachine/caramba/pkg/backend/compute/fusion      0.004s
ok  github.com/theapemachine/caramba/pkg/backend/device/cuda         0.002s
ok  github.com/theapemachine/caramba/pkg/backend/device/metal        0.001s
ok  github.com/theapemachine/caramba/pkg/backend/device/xla          0.001s
```

## Selected benchmark output (linux/arm64, Go 1.26)

```
BenchmarkAllReduce_Sum-4    132193    1639 ns/op   9997.56 MB/s   4192 B/op   2 allocs/op
BenchmarkAllReduce_Mean-4   128790    1928 ns/op   8498.38 MB/s   4192 B/op   2 allocs/op
BenchmarkAllReduce_Max-4    139599    1749 ns/op   9368.24 MB/s   4192 B/op   2 allocs/op
BenchmarkBroadcast_4-4     1352702     177.0 ns/op 23137.47 MB/s    96 B/op   1 allocs/op

BenchmarkBFloat16ToFloat32_1024-4  ~6 µs/op  ~340 MB/s     0 allocs/op
BenchmarkFloat32ToBFloat16_1024-4  ~6 µs/op  ~680 MB/s     0 allocs/op
BenchmarkFloat32ToFloat64_1024-4   ~1.5 µs/op ~2.7 GB/s    0 allocs/op
BenchmarkFloat8E4M3ToFloat32_1024-4  1173 ns/op  873 MB/s   0 allocs/op

BenchmarkBF16_Float32-4              1.676 ns/op
BenchmarkFloat16_Float32-4           1.662 ns/op
BenchmarkFloat8E4M3_FromFloat32-4    1.769 ns/op
BenchmarkFloat8E5M2_FromFloat32-4    1.912 ns/op
```

These are the scalar reference numbers; the SIMD `.s` paths replace
them in a hardware-verified session and the benchmarks here become
the regression bar.

## Phase coverage at end of session

| phase | scope                         | status         |
| ----- | ----------------------------- | -------------- |
| 1     | dtype consolidation           | verified       |
| 2     | SIMD conversion kernels       | scalar verified; SIMD `.s` deferred |
| 3     | HostBackend end-to-end        | verified       |
| 4     | Metal device backend          | skeleton + stub returning ErrNeedsPlatformSetup |
| 5     | CUDA device backend           | skeleton + stub returning ErrNeedsPlatformSetup |
| 6     | XLA device backend            | skeleton + stub returning ErrNeedsPlatformSetup |
| 7     | legacy kill                   | in progress — first compute/runtime/transport slice migrated |
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
| `pkg/backend/device/metal/bridge_darwin.go`   | verified              |
| `pkg/backend/device/metal/bridge_darwin.h`    | verified              |
| `pkg/backend/device/metal/bridge_darwin.m`    | verified              |
| `pkg/backend/device/metal/add_float32.metal`  | verified              |
| `pkg/backend/device/metal/kernels.metallib`   | verified              |
| `pkg/backend/device/metal/generate.go`        | verified              |
| `pkg/backend/device/metal/kernels.go`         | verified              |
| `pkg/backend/device/metal/backend_test.go`    | verified              |
| `pkg/backend/device/metal/internal/metallibgen/main.go` | verified  |
| `pkg/backend/device/metal/internal/metallibgen/main_test.go` | verified |
| `pkg/backend/device/cuda/backend.go`          | verified              |
| `pkg/backend/device/cuda/bridge_stub.go`      | verified              |
| `pkg/backend/device/cuda/bridge_real.go`      | needs-platform-setup  |
| `pkg/backend/device/cuda/backend_test.go`     | verified              |
| `pkg/backend/device/xla/backend.go`           | verified              |
| `pkg/backend/device/xla/bridge_stub.go`       | verified              |
| `pkg/backend/device/xla/backend_test.go`      | verified              |

## Phase 7: legacy kill

In progress. The 2026-05-18 slice migrated IR, executor, CPU runner,
dispatch, orchestrator, network transport, manifest lowering, and
runtime backend output collection to `tensor.Tensor` / `dtype.DType`.

Still pending: legacy downstream files in
`pkg/backend/compute/{metal,cuda,xla}/`,
runtime network/output paths outside the touched slice, documentation,
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
