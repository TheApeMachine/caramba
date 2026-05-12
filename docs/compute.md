# Compute Backends

Caramba implements every backend from scratch. No backend falls back silently to a slower implementation—if a kernel isn't ported, the build fails. This is the core of the platform.

---

## Design Principles

**Explicit tensor ownership.** Tensors are resident in one backend and never implicitly transferred. Moving data between backends requires an explicit `DownloadFloat64` call at a real boundary.

**Hardware-agnostic IR.** The computation graph is expressed once as an IR (`pkg/backend/compute/ir`) and dispatched to whichever runner is selected. The same graph runs on CPU, CUDA, Metal, and XLA.

**No host staging.** Backend kernels keep tensors resident and only download when a real boundary is crossed (e.g., evaluation, checkpoint save). There is no per-operation round-trip to the host.

---

## Runner Interface

All backends implement the same interface:

```go
// pkg/backend/compute/runner/interface.go

type Runner interface {
    // Execute evaluates the graph for the specified output nodes.
    Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error)

    // Location returns the hardware context string.
    Location() tensor.Location

    // Close releases hardware resources.
    Close() error
}
```

`Location` returns a string like `"cpu"`, `"cuda:0"`, `"metal"`, or `"xla:cpu"` that identifies where tensors computed by this runner reside.

---

## CPU Backend

**Package:** `pkg/backend/compute/cpu`

The CPU backend runs on every platform with no external dependencies. It is the reference implementation and the baseline for correctness.

### Pure Go

All operations have a pure Go implementation. No cgo required.

### SIMD / Assembly

Performance-critical kernels are implemented in platform-specific SIMD assembly alongside the Go fallback:

| Architecture | ISA extensions            | File suffix          |
|--------------|---------------------------|----------------------|
| x86-64       | AVX2, FMA, SSE2           | `_amd64.s`           |
| ARM64        | NEON, SVE                 | `_arm64.s`           |

The Go build system selects the correct assembly file automatically based on `GOARCH`. No build tags are required for SIMD—it is always compiled in on supported architectures.

Example: the Markov blanket kernel has dedicated ARM64 assembly:

```
pkg/backend/compute/cpu/operation/markov_blanket/
├── markov_blanket.go         # Go implementation + interface
├── markov_blanket_arm64.go   # ARM64 entrypoint (links asm)
└── markov_blanket_arm64.s    # NEON intrinsics
```

### Resident Tensor Backend

The CPU backend maintains tensors in-memory and exposes them through the `Float64Tensor` interface. Activation, math, and fused matmul+bias(+GELU) kernels operate directly on resident slices.

### Building

```bash
# Standard build (pure Go + SIMD where available)
go build ./pkg/backend/compute/cpu/...

# Tests
go test ./pkg/backend/compute/cpu/...
```

---

## CUDA Backend

**Package:** `pkg/backend/compute/cuda`

**Requirements:** Linux, NVIDIA CUDA toolkit, cgo enabled.

The CUDA backend exposes resident device tensors. All computation stays on-device. Activation, math, and fused linear kernels dispatch CUDA kernels directly.

### Building

```bash
CGO_ENABLED=1 go build -tags "cgo cuda" ./pkg/backend/compute/cuda/...
CGO_ENABLED=1 go test  -tags "cgo cuda" ./pkg/backend/compute/cuda/...
```

The `cuda` build tag gates all cgo imports. Without it, the package compiles to stubs that return an unsupported-backend error at runtime.

---

## Metal Backend

**Package:** `pkg/backend/compute/metal`

**Requirements:** macOS, Xcode Command Line Tools (Metal-capable GPU).

The Metal backend exposes resident `MTLBuffer` tensors. Compute pipelines are compiled from Metal Shading Language sources embedded in the package.

### Building

```bash
# No separate "metal" tag—Darwin + CGO selects the Metal implementation
CGO_ENABLED=1 go build ./pkg/backend/compute/metal/...
CGO_ENABLED=1 go test  ./pkg/backend/compute/metal/...
```

The build constraint is `//go:build darwin && cgo`. On non-Darwin platforms, the package compiles to stubs.

---

## XLA Backend (PJRT)

**Package:** `pkg/backend/compute/xla`

**Requirements:** XLA headers and a PJRT plugin library (CPU or GPU).

XLA is accessed through the PJRT C API. The backend exposes resident PJRT buffers for activation, elementwise math, matmul, and fused matmul+bias(+GELU).

### Environment Setup

```bash
# Required: XLA include directory
export CARAMBA_XLA_INCLUDE_DIR=/path/to/xla/include
export CGO_CPPFLAGS="-I${CARAMBA_XLA_INCLUDE_DIR}"

# CPU plugin (Linux .so or macOS .dylib)
export CARAMBA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so

# GPU plugin (optional, when a GPU PJRT plugin is available)
export CARAMBA_PJRT_GPU_PLUGIN=/path/to/pjrt_c_api_gpu_plugin.so

# Shared fallback (when one plugin serves all platforms)
export CARAMBA_PJRT_PLUGIN=/path/to/pjrt_c_api.so
```

Plugin lookup order:
1. `CARAMBA_PJRT_CPU_PLUGIN` / `CARAMBA_PJRT_GPU_PLUGIN` (platform-specific)
2. `CARAMBA_PJRT_PLUGIN` (shared fallback)
3. Legacy `PJRT_PLUGIN_PATH`

### Building

```bash
CGO_ENABLED=1 go build -tags "cgo xla" ./pkg/backend/compute/xla/...
CGO_ENABLED=1 go test  -tags "cgo xla" ./pkg/backend/compute/xla/...
```

---

## Orchestrator Passes

Before a graph reaches a runner, it passes through the optimizer in `pkg/backend/compute/orchestrator`:

### CSE — Common Subexpression Elimination

Identifies nodes with identical operation types and input sets. Replaces all duplicates with a single node. Eliminates redundant computation that emerges from composed operation blocks.

### DCE — Dead Code Elimination

Removes nodes whose outputs are never consumed by any output node. Cleans up unused branches that result from parameterized templates.

### Fusion

Merges adjacent compatible operations into a single fused kernel. The primary fusion pattern is matmul + bias + activation (e.g., linear + bias + GELU), which eliminates intermediate tensor allocations and improves cache locality.

```
BEFORE fusion:          AFTER fusion:
MatMul                  FusedLinearGELU
  └─▶ BiasAdd     ═══▶  (single kernel)
        └─▶ GELU
```

### Scheduler

Produces a valid topological execution order. Resolves data dependencies and emits an ordered list of nodes for sequential or parallel dispatch.

---

## Tensor Abstraction

```go
// pkg/backend/compute/tensor/tensor.go

type Float64Tensor interface {
    Shape() []int
    DownloadFloat64() ([]float64, error)
    Location() Location
}

type Location string

const (
    LocationCPU    Location = "cpu"
    LocationCUDA   Location = "cuda:0"
    LocationMetal  Location = "metal"
    LocationXLACPU Location = "xla:cpu"
    LocationXLAGPU Location = "xla:gpu"
)
```

Tensors are never implicitly copied. `DownloadFloat64` is the only escape hatch, and it is intentionally named to make the cost visible at the call site.

---

## Adding a New Backend

1. Create a package under `pkg/backend/compute/<name>/`
2. Implement the `runner.Runner` interface
3. Implement `Float64Tensor` for the backend's native tensor type
4. Add build constraints (`//go:build <tag>`) that gate cgo imports
5. Add a `runner_test.go` that runs the full IR test suite against your runner
6. Add build instructions to this document
