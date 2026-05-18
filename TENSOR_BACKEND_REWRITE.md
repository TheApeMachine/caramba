# Tensor Backend Rewrite

This document is the contract for rebuilding Caramba's tensor and backend layer
correctly. It is the source of truth. Every decision below was made with
correctness and performance as the only tiebreakers, per AGENTS.md §10.

If you find ambiguity while implementing, the doc is wrong — patch the doc
first, then code to it. Do not paper over with an inline judgement call.

---

## 1. Why the current layer is wrong

This section lists every concrete defect that motivates the rewrite. Each item
is rooted in a file path and the exact behavior at that location, so there is
no debate about whether the problem exists.

### 1.1 Float64-as-lingua-franca

The `Backend` interface in `pkg/backend/compute/tensor/tensor.go` exposes only
`UploadFloat64(shape, []float64) (Float64Tensor, error)` and
`DownloadFloat64(Float64Tensor) ([]float64, error)`. There is no path to move
non-float64 data into or out of a backend. Every tensor that enters the system
from a real source is upcast to float64 at the boundary:

`pkg/model/weights/safetensors.go` lines 167–225 read F64, F32, F16, BF16,
I64, I32, U32, I16, U16, I8, U8, BOOL and convert all of them to `[]float64`
before any caller sees them. Loading a 70B bf16 model through this path
requires 560 GB of host RAM (8 bytes per parameter × 7×10¹⁰), regardless of
the destination backend's actual storage format.

`pkg/backend/compute/metal/tensor.go` `UploadFloat64` documents on line 85:
"converts host float64 values into resident Metal float32 storage." So the
actual on-device dtype is float32, the on-host dtype is float64, and the
on-disk dtype is bf16/f16/f32 depending on the model. Every weight goes
bf16 → float64 → float32, with two precision changes per parameter and one
silent doubling of memory traffic.

Apple silicon's GPU has dedicated bf16 and fp16 paths. Hopper and Blackwell
have dedicated FP8 paths. None of this hardware is reachable through the
current contract.

### 1.2 Kernel signatures bake in float64

`Float64Tensor` (an interface that embeds `Tensor` and adds
`CloneFloat64() ([]float64, error)`) appears 1,161 times across 98 files.
Every Metal `operation_executor_*.go`, every CPU/CUDA/XLA equivalent, every
test, takes `Float64Tensor` parameters directly and calls `.CloneFloat64()`
in the body. So the float64 assumption is not localized to the backend —
it is encoded into every kernel signature in the codebase.

This is the proximate reason "add bf16" is not a backend swap. Backends are
replaceable; kernels are not.

### 1.3 The host allocator cannot hold real workloads

`pkg/backend/compute/tensor/tensor.go` line 13 hard-codes
`hostArenaFloat64Elements = 8 * 1024 * 1024` — a fixed 64 MiB arena (8 M
float64 elements). A single 4096×4096 fp32 weight tile is 64 MiB; the same
tile stored as float64 (which is what this arena requires) is 128 MiB and
does not fit. A 32 K-context attention scratch matrix is multiples of that.

`HostBackend.allocateArena` returns
`"tensor: host arena exhausted: requested N float64 elements with M reusable of K"`
when the arena fills. There is no growth path, no spillover, no fallback to
heap allocation, no second arena, and no eviction policy. Real workloads hit
this on the first weight upload.

### 1.4 `HostTensor.Float64()` aliases the arena

Line 595 of `tensor.go`: `Float64()` returns the underlying `[]float64`
backing slice with the comment "mutating elements changes the tensor in
place." Two distinct correctness problems follow.

First, any kernel that takes the slice and uses it as scratch corrupts the
tensor for every other consumer. The contract is silently shared-mutable.

Second, `HostTensor.Close()` releases the slot back to the arena free-list
and the arena immediately reuses the space on the next `allocateArena`.
Concurrent `Close()` and `Float64()` is a use-after-free on the arena, since
the `RWMutex` on the tensor only protects field reads, not the lifetime of
the returned slice.

### 1.5 The host allocator serializes all uploads

`HostBackend.mu` is a single `sync.Mutex` guarding the arena, the free-list,
the offset, the live count, and the closed flag. `UploadFloat64`,
`AdoptFloat64`, `releaseArenaTensor`, and `Reset` all hold it exclusively.
Loading a multi-billion-parameter model with N transformer blocks in
parallel serializes through this lock. No streaming, no parallel ingest.

### 1.6 The arena is 8-byte aligned

`make([]float64, hostArenaFloat64Elements)` returns a slice whose backing
array is 8-byte aligned. AVX2 prefers 32-byte alignment, AVX-512 prefers 64,
NEON prefers 16. Every SIMD load through host tensors pays the
unaligned-load penalty on every kernel.

### 1.7 `state.Tensor` is a parallel implementation

`pkg/runtime/state/tensor.go` defines its own `*Tensor` struct holding
`[]float64` plus a `[]int` shape. It does not import
`pkg/backend/compute/tensor` at all. The runtime's `state` registry binds
the type name `"tensor"` to `newTensorFromConfig` (state/state.go line 162),
which allocates `make([]float64, ...)` directly. Consequences:

- The runtime state subsystem has no notion of Location. State tensors
  cannot ever live on Metal/CUDA/XLA. The "Backend-resident variants attach
  through the Bind path" comment in the docstring is aspirational; there is
  no Bind path.
- `Snapshot` serializes as schema `"float64-le-shape-header"` (line 157),
  baking float64 into the wire format used by checkpointing, beam search,
  speculative decoding, and ablation forks.
- Activations, latents, embeddings, and intermediate runtime values are
  allocated outside the arena, outside any memory accounting, with no
  alignment guarantees and no participation in the residency planner.

### 1.8 `dtype/bfloat16.go` uses big-endian byte order

Lines 24, 32 of `pkg/dtype/bfloat16.go` use `binary.BigEndian` for the
`Bytes`/`Decode`/`Encode`/`DecodeFloat32`/`EncodeFloat32` methods on `BF16`.
Safetensors, GGUF, and every hardware target Caramba runs on are
little-endian. The code is dead today (no callers in the repo), but if a
caller ever appears it will corrupt every byte it touches. The methods were
imported from Ollama; whatever invariant they satisfy in Ollama does not
hold here.

### 1.9 The DType enum is wrong

`pkg/backend/compute/tensor/tensor.go` defines `DType` as `{Float64, Float32,
Int64}`. The set the platform actually has to handle, today, is `{F64, F32,
F16, BF16, I64, I32, I16, I8, U32, U16, U8, BOOL}` (safetensors loader, file
already cited). The set it has to handle in any realistic future is the
above plus `{FP8E4M3, FP8E5M2, I4, U16, U64}`. The three-constant enum is a
lie.

### 1.10 The `Tensor.Bytes()` method records a frozen byte count

`HostTensor.bytes` is set at upload time from `shape.Bytes(Float64)` and
never recomputed. Once the new contract permits shape changes (it must, for
KV-cache growth), this becomes stale. The field is the wrong thing to cache.

### 1.11 Snapshot/Restore allocates per call

`state.Tensor.Snapshot` allocates `make([]byte, 4+8*len(shape))` plus
`make([]byte, 8*len(values))` and concatenates with `append` for each call.
Beam search and speculative decoding fork state at every step; this is a
hot allocation path with no buffer reuse.

### 1.12 No quantization support

INT8 inference, INT4 inference, GPTQ, AWQ, FP8 — none are reachable. The
dtype enum does not include them, the kernel API takes Float64Tensor, the
allocator only sizes float64 slots. Inference workloads at scale require at
least INT8 and FP8; this is table stakes for the "real backend" the
codebase is trying to be.

### 1.13 No async / pinned-memory upload

On Apple silicon, host and GPU share physical memory; the correct primitive
is `MTLBuffer` with `MTLResourceStorageModeShared`, which makes "upload" a
metadata operation rather than a copy. Current Metal `UploadFloat64`
copy-converts every byte. On CUDA, pinned host memory (cudaMallocHost) is
required for fast DMA; not present. No async upload, no compute streams.

---

## 2. Target contract

This section defines the contract every implementation must satisfy after
the rewrite. Read this as the API spec; everything in §3 onward is how we
get from §1 to here.

### 2.1 `pkg/dtype` is the canonical source

The `dtype` package owns `DType`, every concrete numeric format, and every
conversion. No other package defines a dtype enum. No other package
hard-codes a numeric format. The `compute/tensor` package's `DType` is
deleted in favor of a `dtype.DType` alias or direct import.

The full enum is:

```go
package dtype

type DType uint8

const (
    Invalid DType = iota

    // Floating point.
    Float64
    Float32
    Float16
    BFloat16
    Float8E4M3
    Float8E5M2

    // Signed integers.
    Int64
    Int32
    Int16
    Int8
    Int4 // packed two-per-byte, little-endian nibble order

    // Unsigned integers.
    Uint64
    Uint32
    Uint16
    Uint8

    // Boolean. Packed eight-per-byte, little-endian bit order.
    Bool

    // Complex.
    Complex64
    Complex128
)
```

Every DType has a `Size()` returning bytes-per-element for unpacked dtypes
and bytes-per-eight-elements (for Bool) or bytes-per-two-elements (for
Int4). A `LogicalElements(bytes int) int` helper inverts this. A
`Parse(name string) (DType, error)` handles the safetensors-style strings
("F16", "BF16", "I8", "BOOL", etc.) and the canonical lowercase names.

Endianness for all serialization formats is little-endian everywhere. The
existing `dtype/bfloat16.go` BigEndian methods are rewritten or deleted; see
§3.3.

### 2.2 `pkg/dtype` numeric types

`dtype.Float16` (existing) and `dtype.BF16` (existing, fixed to LE) remain
the canonical Go-level half-precision types. New types are added:

```go
type Float8E4M3 uint8
type Float8E5M2 uint8
type Int4Pair   uint8 // packs two int4 nibbles
```

Each numeric type provides `Float32() float32`, a `Bits() <underlying>`, and
a constructor from `float32`. FP8 conversion uses the canonical
saturating-on-overflow rounding rule (IEEE 754-2019 round-to-nearest-even,
saturate to ±max-finite on overflow — match the H100/B200 hardware
behavior). All deviations from IEEE default rounding require explicit
documentation in the type's godoc.

### 2.3 `pkg/backend/compute/tensor` interfaces

The package keeps its name but is rebuilt around dtype-aware contracts. The
`Tensor` interface becomes:

```go
package tensor

import "github.com/theapemachine/caramba/pkg/dtype"

type Tensor interface {
    Shape()    Shape
    DType()    dtype.DType
    Layout()   Layout       // see §2.16; LayoutDense for non-sparse
    Location() Location
    Len() int               // logical element count
    Bytes() int             // storage bytes, computed from Shape × DType
    Close() error

    // Zero-copy view primitives. See §2.5.
    Slice(start, length int) (Tensor, error)
    Reshape(dims []int)      (Tensor, error)

    // Native views: return a typed slice aliasing the tensor's storage
    // only if storage dtype matches. Otherwise return ErrDTypeMismatch.
    // The returned slice is valid until Close, Sync, the next mutating
    // method on this tensor, or the start of any async backend operation
    // touching this tensor. Mutation through the slice is permitted and
    // affects storage in place. Callers must not retain the slice past
    // its validity window. Concurrent host-side mutation while the tensor
    // is in StatePending (see Sync/Ready) returns ErrTensorInTransit.
    Float64Native()    ([]float64, error)
    Float32Native()    ([]float32, error)
    Float16Native()    ([]dtype.Float16, error)
    BFloat16Native()   ([]dtype.BF16, error)
    Float8E4M3Native() ([]dtype.Float8E4M3, error)
    Float8E5M2Native() ([]dtype.Float8E5M2, error)
    Int64Native()      ([]int64, error)
    Int32Native()      ([]int32, error)
    Int16Native()      ([]int16, error)
    Int8Native()       ([]int8, error)
    Uint64Native()     ([]uint64, error)
    Uint32Native()     ([]uint32, error)
    Uint16Native()     ([]uint16, error)
    Uint8Native()      ([]uint8, error)
    BoolNative()       (BitVector, error)
    Int4Native()       (Int4Vector, error)

    // RawBytes returns the storage as bytes plus its dtype. Always
    // succeeds for any on-host tensor. For device tensors, it materializes
    // a host copy. Caller owns the returned slice.
    RawBytes() (dtype.DType, []byte, error)

    // Lifecycle state machine. See §2.8 and §2.13.
    State() State                       // StateReady | StatePending | StateMutating | StateClosed
    Sync(ctx context.Context) error     // block until StateReady
    Ready() <-chan struct{}             // closes when state transitions to Ready

    // Autograd. See §2.18.
    RequiresGrad() bool
    SetRequiresGrad(yes bool) error
    Grad()   (Tensor, error)
    GradFn() GradFn
}
```

The `*Native` accessors are zero-copy when storage dtype matches and never
convert. Conversion is an explicit operation, exposed by the standalone
helper package `pkg/dtype/convert`:

```go
func ToFloat32(input Tensor) ([]float32, error)
func ToBFloat16(input Tensor) ([]dtype.BF16, error)
// ...one per target dtype.
```

This makes precision changes auditable. There is no method anywhere whose
implementation silently converts. If a caller wants float64 from a bf16
tensor, the call site reads `convert.ToFloat64(tensor)`, which is impossible
to miss in code review.

`Float64Tensor` is deleted. `CloneFloat64` is deleted. `Float64From`,
`MustFloat64From`, `MustCloneFloat64` are deleted or rewritten as wrappers
around the new contract.

### 2.4 `BitVector` and `Int4Vector`

```go
type BitVector struct {
    data []byte
    length int
}

func (vector BitVector) Len() int
func (vector BitVector) Get(index int) bool
func (vector BitVector) Set(index int, value bool)
func (vector BitVector) Bytes() []byte // underlying storage, len = (length + 7) / 8

type Int4Vector struct {
    data []dtype.Int4Pair
    length int
}

func (vector Int4Vector) Len() int
func (vector Int4Vector) Get(index int) int8 // sign-extended
func (vector Int4Vector) Set(index int, value int8)
func (vector Int4Vector) Bytes() []byte // underlying storage, len = (length + 1) / 2
```

Both types live in the `tensor` package. The packing order is
little-endian: bit 0 / nibble 0 occupies the LSB of byte 0.

### 2.5 `Shape`

`Shape` keeps validation but loses `Bytes(DType)` because `Tensor.Bytes()`
now computes storage size from `Shape × DType` directly. Storage is
strictly contiguous row-major. General strides are not supported — see
§5.6 for why.

```go
type Shape struct {
    dims     []int
    elements int
    valid    bool
}

func NewShape(dims []int) (Shape, error)
func (shape Shape) Dims() []int
func (shape Shape) Len()  int
func (shape Shape) Equal(other Shape) bool
```

Operations that look stride-like are handled explicitly. Reshape that
preserves element order (changes only how dims partition the flat
sequence) is metadata-only and free; reshape that would change element
order is not a reshape, it is a transpose, and transposes materialize.
The single zero-copy view primitive is offset+length slicing:

```go
// Slice returns a zero-copy 1-D subview of the tensor's contiguous
// storage starting at element offset start, of length elements.
// Storage is shared with the parent; the subview increments the
// parent's reader counter and must be Closed before the parent.
// The returned tensor has Shape [length] and the parent's DType.
// For multi-dimensional sub-tensors, compute the offset manually
// (start = batch_index * batch_stride) and Reshape the result.
Slice(start, length int) (Tensor, error)
Reshape(dims []int) (Tensor, error) // metadata-only; product must match
```

Slice is the safetensors-mmap-into-pieces primitive and the KV-cache
active-window primitive. Reshape is metadata-only on contiguous storage.
Transpose, Permute, Broadcast, and any non-contiguous rearrangement
materialize through `tensor.Contiguous(rearrangement)` which returns a
freshly allocated contiguous tensor.

### 2.6 `Backend`

```go
type Backend interface {
    Location() Location
    SupportedDTypes()  []dtype.DType
    SupportedLayouts() []Layout
    Capabilities() Capabilities

    // Upload takes raw bytes plus dtype. The backend chooses its on-device
    // storage dtype, which may differ from the input dtype. The returned
    // tensor's DType() reports the on-device dtype, not the input dtype.
    // SupportedDTypes lists which input dtypes are accepted without
    // conversion at the boundary; others return ErrDTypeUnsupported and
    // the caller must convert explicitly first.
    Upload(shape Shape, sourceDType dtype.DType, bytes []byte) (Tensor, error)

    // UploadSparse uploads a sparse tensor in the layout indicated by
    // layout. Indices follow the per-layout convention in §2.16.
    UploadSparse(
        shape Shape,
        valueDType dtype.DType,
        layout Layout,
        values []byte,
        indices []SparseIndex,
    ) (SparseTensor, error)

    // Download materializes a host copy of the tensor's storage. The
    // returned dtype matches the device-side dtype.
    Download(tensor Tensor) (dtype.DType, []byte, error)

    Close() error
}

type Capabilities struct {
    MaxBytes         int64           // total storage budget
    SupportsAsync    bool
    SupportsSparse   bool            // see §2.16
    SupportsAutograd bool            // see §2.18
    NativeAlignment  int             // bytes, for aligned-load throughput
    NUMANodes        int             // host-only; 0 elsewhere; see §2.7
}
```

`UploadFloat64`/`DownloadFloat64` are deleted. Convenience wrappers
(`UploadFloat32`, `UploadBFloat16`, etc.) live in `pkg/backend/compute/tensor/upload.go`
as free functions over `Backend`, not interface methods.

### 2.7 Host allocator

The `HostBackend` is rewritten. The 64 MiB fixed-size float64 arena is
deleted. The replacement is a three-tier allocator with explicit
no-zero semantics and NUMA awareness.

**Tier 1 — small (< 1 MiB).** A per-P sharded slab allocator, *not*
`sync.Pool`. `sync.Pool` would have been the obvious choice but it is
wrong here: the Go runtime drains `sync.Pool` on every GC cycle (the
victim cache survives at most one cycle), which means precisely when
memory pressure spikes — exactly when pooling should save us — the
pool empties and reallocations happen anyway. For a research substrate
that wants predictable latency, that is unacceptable. The slab
allocator in `pkg/backend/compute/tensor/slab.go` keeps a per-P
free-list of power-of-two-sized blocks, backed by a chunk of the same
mmap region Tier 2 uses. Allocation: pop from local list, fall back to
the global list, fall back to mmap a new chunk. Free: push to local
list. No GC interaction; the runtime never touches these blocks.
Concurrent access uses `sync/atomic` CAS on the per-P list head.

**Tier 2 — medium (1 MiB – 1 GiB).** Per-size-class free-lists backed
by anonymous `mmap`. Allocations bypass the Go heap (and the GC) and
are tracked by an explicit reference. On `Close`, the buffer is
`MADV_DONTNEED`'d (the kernel reclaims the physical pages without
unmapping the virtual range) and returned to the free-list. Subsequent
re-acquisition is a virtual remap with no zeroing — the kernel maps
zero-pages on first touch only if read before write, and the contract
forbids reading-before-writing on freshly handed buffers.

**Tier 3 — large (≥ 1 GiB).** Direct anonymous `mmap` with
`MADV_HUGEPAGE` on Linux (2 MiB / 1 GiB pages reduce TLB pressure on
the multi-billion-parameter weight loads this tier handles). On Darwin,
`mmap` with `VM_FLAGS_SUPERPAGE_SIZE_ANY`. Buffers are never returned
to the OS; freed buffers stay on the free-list indefinitely because
weight loads recur. Allocator overhead amortizes across the model's
lifetime.

**No-zero default.** Tiers 2 and 3 hand out memory whose contents are
indeterminate (zero on first map, the previous tenant's bytes after
recycle). Callers that need zero-initialized memory call
`tensor.NewZeroed(shape, dtype)`; the default `tensor.New(shape, dtype)`
returns uninitialized storage. Every `Backend.Upload` path overwrites
every byte before any reader sees it, so the default is correct for the
hot path. The escape hatch exists so RMS norm and similar zero-init
patterns are explicit at the call site.

**Alignment.** Every allocation returns a base pointer aligned to 64
bytes (AVX-512 width). The allocator over-allocates by 64 bytes and
slices to the first aligned boundary; the raw pointer is retained for
free-list bookkeeping. The aligned base is asserted at the boundary
with `if uintptr(unsafe.Pointer(&buf[0])) % 64 != 0 { panic }`.

**NUMA awareness (host only).** On Linux, the allocator queries
`numa_available()` and `numa_num_configured_nodes()` at init. Each
tensor allocation accepts an optional NUMA node parameter; the default
is the current goroutine's preferred node via
`get_mempolicy(MPOL_F_NODE)`. Allocations honor `mbind(MPOL_BIND, node)`
so pages are placed on the requested node and stay there. Kernels
launched against a NUMA-pinned tensor inherit the affinity through
`pthread_setaffinity_np` on the worker thread. On Darwin (single NUMA
node per process on Apple silicon and current Intel Macs), all
NUMA-related calls are no-ops; `Capabilities.NUMANodes` returns 1.

**Bump arenas for forward-pass scratch (opt-in).** The executor can
hand a `*tensor.Arena` to its scope. Arena tensors share a single mmap'd
region (Tier 2/3 backed) and are handed out by bump-allocation. Their
`Close()` is a no-op; the arena is `Reset()` wholesale at step
boundaries, which `MADV_DONTNEED`s the region without returning it to
the free-list. Reset invalidates every outstanding tensor handle; the
reader counter (§2.8) detects any post-reset access and panics.

**Concurrency.** All three tiers are lock-free for the hot path. Tier 1
uses `sync.Pool`'s internal P-sharded structure. Tiers 2 and 3 use
per-size-class free-lists guarded by `sync/atomic` CAS on the head
pointer. Concurrent uploads from N goroutines scale linearly with N
until memory bandwidth saturates.

### 2.8 Aliasing rules and lifecycle state machine

Native accessors return plain typed slices that alias the tensor's
storage. There is no `View[T]` wrapper, no `Release()` call, and no
reader counter — that earlier design was a Rust/C++ idiom forced into
Go and would have leaked the moment a kernel forgot a `defer` or a
caller recovered a panic mid-loop. The `release func()` closure would
also have heap-allocated on every native-view call, defeating escape
analysis and creating massive GC churn in the hot path.

The aliasing contract is enforced instead by an explicit tensor state
machine that every backend implementation honors:

```go
type State uint8

const (
    StateReady    State = iota  // host-mutable, view-accessible, can be uploaded
    StatePending              // async op in flight; host views are forbidden
    StateMutating             // a native view is outstanding; uploads forbidden
    StateClosed               // Close has run; all operations error
)
```

Transitions:

- `StateReady → StatePending`: triggered by `Backend.UploadAsync`,
  inter-backend migration, or any async device-side op that reads or
  writes this tensor. `StatePending → StateReady` on completion.
- `StateReady → StateMutating`: triggered by acquiring a native view
  (`Float32Native()` etc.) or by the in-place math kernels in §2.11.
  The transition is rejected with `ErrTensorInTransit` if the tensor
  is in `StatePending`. The tensor remains `StateMutating` until the
  caller calls `tensor.ReleaseMutation()` or all outstanding native
  slices have been observed-released via the GC-pinned tracker (see
  below).
- Any state → `StateClosed`: by `Close()`. Close blocks until the
  tensor is `StateReady`, draining any in-flight async ops with their
  context. If `StateMutating` is active when Close is called, Close
  returns `ErrTensorMutating`; the caller must release first.

Native view validity:

- The returned slice is valid until `Close`, `Sync`, the next mutating
  call on this tensor, or the next async backend op that touches this
  tensor.
- Mutation through the slice is permitted.
- Concurrent access across goroutines is the caller's responsibility.
  The race detector catches data races; lifetime errors are a
  programmer responsibility and are documented per backend.

Mutation tracking without `Release()`. To detect "tensor still in
`StateMutating` when Close arrives" without forcing `defer release()`
on every caller, the implementation pins each outstanding native
slice's backing-array pointer with `runtime.AddCleanup` (Go 1.24+).
When the slice is GC'd, the cleanup decrements an internal counter
and transitions the tensor back to `StateReady` if the counter hits
zero. This is best-effort: kernels that immediately drop the slice
after their inner loop see fast transitions; kernels that escape the
slice to the heap pay GC-cycle latency. It is correctness-preserving
either way — Close blocks until the counter drains or the context
expires.

Async upload (§2.13) and view acquisition (§2.8) thus cannot race:
the state machine forces one or the other. The host CPU mutating a
slice while DMA is in flight is impossible because acquiring the
slice would have failed with `ErrTensorInTransit`.

### 2.9 `Set` is removed

The current `HostTensor.Set(shape, values)` allows a tensor to change shape
and content in-place. This breaks every cached metadata invariant
(`bytes`, `Shape`, `Strides`) and makes lifetime reasoning much harder.
Replacement: callers that need a new shape allocate a new tensor.
In-place math goes through dedicated kernels that touch values but not
shape.

### 2.10 Snapshot / Restore

`state.Tensor` is rewritten to wrap a `tensor.Tensor`. Its snapshot
schema becomes:

```
schema = "tensor.v1"
payload layout:
    u8  schema_version                 = 1
    u8  dtype                          = dtype.DType value
    u8  layout                         = tensor.Layout value (§2.16)
    u8  reserved                       = 0
    u32 rank
    rank × u64 dimension
    u64 byte_length
    byte_length × u8 raw storage bytes
    if layout == LayoutSparseCSR:
        u64 nnz
        (rank-aware index payload, see §2.16)
```

The schema string is `"tensor.v1"`. Bumping schema versions is required for
any layout change. Restore validates schema version before parsing.

Snapshot payload allocation reuses a per-state `[]byte` buffer where
possible, sized to `8 + 8 + 4 + 8*rank*2 + 8 + byteLen` on first use.
Beam search forks reuse the buffer across snapshots in the same forward
pass; cross-pass snapshots reallocate.

### 2.11 Kernel signatures

Kernels accept `tensor.Tensor` and dispatch on `DType()`. The
implementation pattern:

```go
func (matmul *MatMul) Run(
    output, lhs, rhs tensor.Tensor,
) error {
    dtype := lhs.DType()

    if rhs.DType() != dtype || output.DType() != dtype {
        return errMixedDType(output, lhs, rhs)
    }

    switch dtype {
    case dtype.Float32:
        return matmul.runFloat32(output, lhs, rhs)
    case dtype.BFloat16:
        return matmul.runBFloat16(output, lhs, rhs)
    case dtype.Float64:
        return matmul.runFloat64(output, lhs, rhs)
    }

    return errUnsupportedDType(dtype)
}
```

Mixed-dtype kernels (e.g. bf16 weights × fp32 activations with fp32
accumulation, as in standard transformer matmul) are first-class. They
declare their accepted dtype combinations and reject others with
`ErrDTypeCombinationUnsupported`. Implicit upcasting is forbidden — the
caller dispatches a mixed-dtype kernel by name.

Per AGENTS.md §1, every kernel has parity tests at
`N ∈ {1, 7, 64, 1024, 8192}` against the scalar Go reference for that
dtype, with tight ULP bounds. No widening of tolerance to absorb
conversion error. If a kernel produces results outside the bound, the
kernel is wrong; the test does not change.

### 2.12 Metal-specific contract

On Apple silicon, host and GPU share physical memory. The Metal backend
allocates all tensor storage via `MTLBuffer` with
`MTLResourceStorageModeShared`, and the host view of that buffer is the
same byte range as the device view. Upload is metadata; the bytes are not
copied. The Metal backend's `SupportedDTypes` returns
`{Float32, BFloat16, Float16, Int8, Int4, Bool}` initially; FP8 paths land
as M3/M4 Metal Performance Primitives expose them.

The Metal backend's `NativeAlignment` is 256 bytes (Apple's recommended
buffer alignment for vector loads).

### 2.13 CUDA-specific contract

Host buffers for CUDA upload use `cudaMallocHost` (page-locked) to enable
DMA. Device storage uses standard `cudaMalloc`. Upload becomes an async
copy on a dedicated upload stream; the returned tensor is in
`StatePending` and the state machine in §2.8 prevents host-side view
acquisition until the upload event has fired and the tensor transitions
to `StateReady`. Kernels that consume the tensor on the same stream as
the upload are not blocked — CUDA stream ordering ensures the kernel
runs after the copy without an explicit sync. Kernels on a different
stream call `Tensor.Sync(ctx)` first.

`SupportedDTypes` returns
`{Float32, BFloat16, Float16, Float8E4M3, Float8E5M2, Int8, Int4, Bool}`
on H100/B200, `{Float32, BFloat16, Float16, Int8, Int4, Bool}` on Ampere.
The exact set is queried from the device's compute capability at
backend init.

`NativeAlignment` is 128 bytes.

### 2.14 XLA-specific contract

XLA tensors are HLO buffers. The XLA backend translates Caramba `Tensor`
operations to HLO and submits via the XLA runtime. `SupportedDTypes` is
the full XLA-native set; conversion happens inside XLA when shapes admit.

`NativeAlignment` follows the platform (TPU: 128 bytes; XLA-GPU: 128 bytes).

### 2.15 Network transport

`pkg/network/transport` serializes tensors using the Snapshot v1 wire
format from §2.10. The transport carries
`(location, dtype, layout, shape, bytes [+ sparse indices])`. On receive,
the remote backend runs `Upload(shape, dtype, layout, bytes)`; no
float64 round-trip.

### 2.16 Sparse tensors

Sparsity is a property of storage layout, not of element dtype. A new
enum lives alongside `Tensor`:

```go
type Layout uint8

const (
    LayoutDense       Layout = iota
    LayoutSparseCSR          // Compressed Sparse Row
    LayoutSparseCSC          // Compressed Sparse Column
    LayoutSparseCOO          // Coordinate
    LayoutSparseBSR          // Block Sparse Row (block_size in metadata)
)
```

`Tensor.Layout() Layout` is added to the interface in §2.3. Dense
tensors return `LayoutDense`; sparse tensors return the active layout.
Sparse tensors expose layout-specific accessors:

```go
type SparseTensor interface {
    Tensor
    NNZ() int                              // non-zero element count
    Values() (Tensor, error)               // dense 1-D tensor of nonzeros
    Indices() ([]SparseIndex, error)       // layout-specific index sets
    BlockSize() (rows, cols int, ok bool)  // LayoutSparseBSR only
}

type SparseIndex struct {
    Name  string // "row_ptr", "col_idx", "row_idx", etc.
    Data  Tensor // typically Int32 or Int64
}
```

The `Backend.Upload` extension for sparse is a separate method to
keep the dense fast-path simple:

```go
UploadSparse(
    shape Shape,
    valueDType dtype.DType,
    layout Layout,
    values []byte,
    indices []SparseIndex,
) (SparseTensor, error)
```

CSR format: indices = [`{Name: "row_ptr", Data: int32[rows+1]}`,
`{Name: "col_idx", Data: int32[nnz]}`].
CSC mirrors CSR with columns. COO carries one index tensor per
dimension. BSR adds a block-size descriptor to the tensor metadata.

Sparse kernels live alongside dense kernels and dispatch on
`tensor.Layout()` after dispatching on `tensor.DType()`. The kernel
dispatch table in §2.11 is extended to a two-level dispatch:
`(layout, dtype)`.

### 2.17 Distributed / sharded tensors

A logical tensor partitioned across multiple physical tensors on
multiple backends or nodes is a `DistributedTensor`. The partitioning
is described by a `ShardingMesh` (a topology of devices) and a
`ShardingSpec` (how each dim of the logical tensor maps to the mesh).

```go
type ShardingMesh struct {
    Devices []Location  // mesh nodes; len = product(MeshShape)
    Shape   []int       // mesh dimensions (e.g. [4, 2] = 4x2 mesh)
    AxisNames []string  // optional axis labels ("data", "model")
}

type ShardingSpec struct {
    PerDim []DimSharding // len = tensor rank
}

type DimSharding struct {
    Replicated bool   // value replicated on every device along this axis
    ShardAxis  int    // mesh axis this dim is sharded along; -1 if replicated
}

type DistributedTensor interface {
    LogicalShape() Shape
    DType() dtype.DType
    Layout() Layout
    Mesh() ShardingMesh
    Sharding() ShardingSpec
    Shards() []Tensor          // one per mesh device; index is mesh-flatten
    LocalShard() (Tensor, error) // shard for this process's device
    Close() error
}
```

Each `Tensor` returned by `Shards()` is a regular Tensor on its own
Location. Collective operations (all-reduce, all-gather, reduce-scatter,
broadcast) live in `pkg/backend/compute/collective` and are dispatched
by op-graph nodes that the kernel layer recognizes; kernels themselves
do not call collectives directly. Cross-shard communication uses
NCCL/RCCL on CUDA, MPS-side ring all-reduce on Metal (when multiple
GPUs are present in a single Mac Studio / Mac Pro), and the
`pkg/network/transport` framing on host.

### 2.18 Autograd / tape recording

The compute layer becomes forward-and-backward. Every operation that
participates in autograd records a gradient function. The default is
opt-in: tensors that do not need gradients pay no overhead.

```go
type Tensor interface {
    // ...existing methods...
    RequiresGrad() bool
    Grad() (Tensor, error)
    GradFn() GradFn
    SetRequiresGrad(yes bool) error
}

type GradFn interface {
    // Backward computes the gradient with respect to each input given
    // the upstream gradient. Returns one gradient tensor per input, in
    // the same order the inputs were supplied to the forward op.
    Backward(ctx context.Context, upstream Tensor) ([]Tensor, error)
    Inputs() []Tensor
}

type Tape struct {
    // Records ops in order. Backward walks in reverse and accumulates.
}

func NewTape() *Tape
func (tape *Tape) Record(op GradFn)
func (tape *Tape) Backward(ctx context.Context, output Tensor) error
```

Forward operations check `RequiresGrad()` on their inputs and record a
`GradFn` on the tape iff any input requires grad. The output tensor
inherits `RequiresGrad = true` in that case. Operations whose backward
is not implemented return `ErrBackwardNotImplemented` at record time,
not at backward time, so missing coverage is caught early.

Gradient accumulation: `Tensor.Grad()` returns the accumulated gradient.
The first backward pass writes; subsequent passes that don't call
`tensor.ZeroGrad()` add. This matches PyTorch's accumulation semantics.

Mixed-precision: when a forward op upcasts (e.g. bf16 matmul accumulating
in fp32), the backward uses the same upcast convention. The kernel is
responsible for the dtype combination; the autograd layer is dtype-blind.

### 2.19 Graph-level fusion

The orchestrator's CSE, DCE, and fusion passes are made dtype-aware,
layout-aware, and sharding-aware. Two ops can be CSE'd only if their
output `(DType, Layout, ShardingSpec)` triples match exactly. Fusion
across a dtype-changing op (cast, convert) is forbidden unless the
resulting numeric path is reasoned about and added to the fusion
catalog with a parity test.

Fusion catalog entries are explicit: each entry names the source ops,
the fused op, the dtype combination, and the parity bound. No
opportunistic fusion. The orchestrator's pass declares supported
fusions; unrecognized op sequences pass through unfused.

The orchestrator's tests are extended to cover every fusion entry with
a parity check against unfused execution at
`N ∈ {1, 7, 64, 1024, 8192}`.

### 2.20 Conversion kernels

Every dtype↔dtype conversion is a first-class kernel in
`pkg/backend/compute/convert`. These are not Go scalar loops disguised
as helpers — they are full members of the kernel family with the same
five-host-ISA requirement as every other SIMD operation (scalar Go +
AVX-512 + AVX2 + SSE2 + NEON), plus Metal, CUDA, and XLA paths. The
required pairs at minimum:

- `bf16 ↔ f32`, `f16 ↔ f32`, `f32 ↔ f64`
- `bf16 ↔ f16`
- `fp8e4m3 ↔ f32`, `fp8e5m2 ↔ f32`
- `int8 ↔ f32`, `int4 ↔ f32` (dequant; quant is a separate parameterized kernel)
- `f64 → f32` and inverse for the safetensors loader's legacy path
- Identity dtypes pass through without copy

Throughput requirement: at GiB/s on the host, not MiB/s. AVX-512
`vcvtne2ps2bf16` and `vcvtbf162ps` are the bf16 path on amd64; NEON's
`bfcvtn` is the arm64 path. FP8 has no native amd64 SIMD; the kernel
uses AVX-512 byte shuffles plus a small LUT to hit ≥ 10 GiB/s for
upcasts, which is sufficient to keep weight-load bandwidth-bound rather
than conversion-bound. Conversion kernels carry their own parity tests
at `N ∈ {1, 7, 64, 1024, 8192}` against the canonical reference, with
ULP bounds matched to the target dtype.

The `dtype/convert` package surface from §2.3 wraps these kernels as
free functions over `tensor.Tensor`; the helper picks the right
kernel for the input and output dtypes. There is no "Go scalar loop"
in this package — every helper dispatches to a `convert` kernel.

---

## 3. Migration plan

The work is sequenced so each phase has a clean Definition of Done and the
codebase compiles and tests pass at the end of each phase. No phase is
"minimal version now, real version later" — each is a complete unit.

**Sequencing rationale.** The earlier draft staged the migration as
"widen the kernel API in Phase 2, then make storage dtype-native in
Phase 4." That created a valley-of-death where the bridge between the
two phases ran every operation through silent bf16 → f64 → bf16 → f64
round trips — main would have been ~2× slower for the duration. The
revised sequencing migrates each backend end-to-end (interface widening
+ dtype-native storage + state machine + allocator/async-as-applicable)
in a single phase per backend. During those per-backend phases the
*legacy* `Float64Tensor` path is preserved unchanged on already-migrated
backends so kernels keep compiling against `Float64Tensor` until every
backend is ready and the final kill-the-bridge phase runs the sed pass
and deletes the legacy API. Main never regresses below baseline.

### 3.1 Phase 1: dtype consolidation

`pkg/dtype` becomes the canonical source. Concrete tasks:

3.1.1 Add `dtype.DType` enum per §2.1, with `Size()`, `LogicalElements()`,
      `Name()`, `Parse()`, and a `String()` matching safetensors casing
      (uppercase: F32, BF16, I8). Include every constant in §2.1, even the
      ones whose conversion machinery lands later.

3.1.2 Add `dtype.Float8E4M3` and `dtype.Float8E5M2` as `uint8` newtypes,
      with `Float32()` conversion (saturating round-to-nearest-even) and
      `FromFloat32()` constructors. Parity tests against a reference fp8
      implementation in a comment block.

3.1.3 Add `dtype.Int4Pair` as `uint8` newtype with `Lo()` / `Hi()` /
      `WithLo()` / `WithHi()` accessors, each returning sign-extended
      `int8`.

3.1.4 Fix `dtype/bfloat16.go`: change every `binary.BigEndian` to
      `binary.LittleEndian`. Remove dead methods if they cannot be
      rewritten cleanly. Document the package as little-endian.

3.1.5 Add a scalar-Go-only `dtype/convert` package with
      `ToFloat64(Tensor)`, `ToFloat32(Tensor)`, etc. This is the
      correctness-only conversion path used by the legacy bridge while
      Phases 2–7 are in flight. Perf-critical SIMD conversions land in
      Phase 2; the package surface stays the same so Phase-2 SIMD
      kernels replace the scalar bodies without changing call sites.

3.1.6 Tests for every dtype's `Size()`, `Parse()` round-trip, FP8
      Float32 round-trip with bound-checked precision, and BF16 LE
      serialization round-trip.

Definition of Done for Phase 1: `pkg/dtype` exposes the full enum;
FP8 types compile and round-trip within representable range; BF16
serializes little-endian; the package is purely additive (no callers
have changed yet).

### 3.2 Phase 2: SIMD conversion kernels

`pkg/backend/compute/convert` lands as a first-class kernel family.
Per §2.20, every conversion ships scalar Go + AVX-512 + AVX2 + SSE2 +
NEON. No ISA aliasing. Throughput requirement: bandwidth-bound on
host (≥ 10 GiB/s for f32-width conversions on amd64 / arm64).

3.2.1 `bf16 ↔ f32`: AVX-512 uses `vcvtne2ps2bf16` and `vcvtbf162ps`;
      AVX2 uses the manual exponent-shift trick (mantissa truncation
      with round-to-nearest-even); SSE2 scalar-per-element fallback;
      NEON uses `bfcvt` / `bfcvtn` on armv8.6-A+, manual shift below.

3.2.2 `f16 ↔ f32`: AVX-512 / AVX2 use `vcvtph2ps` / `vcvtps2ph` (F16C);
      SSE2 manual; NEON uses `fcvt` / `fcvtn`.

3.2.3 `f32 ↔ f64`: trivial widen/narrow on all ISAs.

3.2.4 `bf16 ↔ f16`: chain via f32 internally; SIMD-fused on AVX-512
      when both intrinsics are available in one register window.

3.2.5 `fp8e4m3 ↔ f32` and `fp8e5m2 ↔ f32`: no native amd64 SIMD on
      current hardware; the kernel uses AVX-512 byte shuffles plus a
      256-entry LUT in `zmm` for upcast (≥ 10 GiB/s achievable);
      downcast uses saturating arithmetic. NEON: same LUT pattern
      with `tbl` instructions.

3.2.6 `int8 ↔ f32` and `int4 ↔ f32` (dequant only — quant lives with
      the quantization kernels in §3.8): straightforward widen/scale.
      The int4 path uses AVX-512 `vpunpckhbw`/`vpunpcklbw` to unpack
      nibbles into bytes before widening.

3.2.7 Rewrite `pkg/dtype/convert`'s Phase-1 scalar functions to
      dispatch to the SIMD kernels in this package. The package
      surface stays identical.

3.2.8 Rewrite `pkg/model/weights/safetensors.go` to dispatch through
      `convert` instead of inlining its own scalar conversion. This is
      a self-contained change because the loader's output dtype
      becomes the source dtype on disk; downstream `UploadFloat64`
      callers continue to receive `[]float64` via
      `convert.ToFloat64(bytes, sourceDType)`.

3.2.9 Parity tests at `N ∈ {1, 7, 64, 1024, 8192}` per ISA per dtype
      pair. Tight ULP bounds matched to the target dtype's standard.
      Benchmarks pasted.

Definition of Done for Phase 2: every conversion has five-ISA-variant
kernels passing parity at the four corners of `N`; weight-load
benchmark on safetensors-loaded model improves by the
ratio-of-bytes between source dtype and float64 (e.g. ~4× faster
loading a bf16 model than baseline).

### 3.3 Phase 3: HostBackend end-to-end

The Host backend is fully migrated to dtype-native storage with the
new tiered allocator, state machine, and aliasing rules — in one
phase. The legacy `Float64Tensor` / `UploadFloat64` path remains in
place; the new `Tensor` / `Upload(shape, dtype, bytes)` path lives
alongside it. Kernels still compile because `Float64Tensor` is
unchanged.

3.3.1 Implement the tiered allocator per §2.7: Tier 1 sharded slab
      (per-P free-lists, mmap-backed, no `sync.Pool`); Tier 2 mmap +
      `MADV_DONTNEED`; Tier 3 mmap + `MADV_HUGEPAGE`. Files:
      `pkg/backend/compute/tensor/slab.go`,
      `pkg/backend/compute/tensor/mmap_medium.go`,
      `pkg/backend/compute/tensor/mmap_large.go`. Each tier is its
      own file under the size rule.

3.3.2 `tensor.New(shape, dtype) (Tensor, error)` (uninitialized) and
      `tensor.NewZeroed(shape, dtype) (Tensor, error)` (explicit
      zero). Every `Upload` path uses `tensor.New`.

3.3.3 NUMA support: `pkg/backend/compute/tensor/numa_linux.go` (cgo
      to libnuma); `numa_darwin.go` is a single-node no-op.

3.3.4 New `HostTensor` implementation with dtype-native storage. The
      backing buffer is `[]byte` sized to `Shape × DType`; the
      `*Native` accessors return typed slice headers built from the
      buffer with `unsafe.Slice`. State machine per §2.8 enforced
      via `atomic.Uint32` State.

3.3.5 The old `HostTensor` (float64 backing array, 64 MiB arena,
      `Float64Tensor` implementation) remains until Phase 7 deletes
      it. `HostBackend.UploadFloat64` keeps returning the old type
      so existing kernels keep compiling.

3.3.6 New methods on `HostBackend`: `Upload(shape, dtype, bytes)
      (Tensor, error)`, `Download(tensor) (dtype, []byte, error)`,
      `UploadSparse(...)` (the sparse path actually lands its
      implementation in Phase 9 — Phase 3 stubs it returning
      `ErrLayoutUnsupported`).

3.3.7 `tensor.Arena` bump allocator backed by Tier 2/3. Opt-in.

3.3.8 Benchmark: allocation throughput at
      `{64 B, 1 KiB, 64 KiB, 1 MiB, 64 MiB, 1 GiB}` × thread counts
      `{1, GOMAXPROCS, 2 × GOMAXPROCS}` vs. old arena and vs. raw
      `make`. Conversion bandwidth verification: an end-to-end bf16
      weight upload through the new path runs at ≥ 50% of the memcpy
      bandwidth of the system.

3.3.9 NUMA verification on multi-node hardware: local-NUMA bandwidth
      ≥ 2× cross-NUMA bandwidth on a 1 GiB tensor. Skip on
      single-node hosts.

Definition of Done for Phase 3: HostBackend has both legacy and new
paths; new path stores native dtypes; allocator passes its benchmarks;
state machine enforced by tests that attempt forbidden transitions
and verify error returns; existing test suite (which still uses the
legacy path) passes unchanged.

### 3.4 Phase 4: Metal end-to-end

3.4.1 Rewrite Metal tensor storage to use `MTLBuffer` with
      `MTLResourceStorageModeShared`. Upload becomes zero-copy on
      Apple silicon — `copy(buffer.Contents(), sourceBytes)` plus a
      `DidModifyRange` notification, no internal float64
      intermediate.

3.4.2 `Backend.Upload(shape, dtype, bytes)` honors source dtype.
      Native Metal dtypes (`Float32`, `BFloat16`, `Float16`, `Int8`,
      `Int4`, `Bool` per §2.12) store as-is. Other dtypes convert to
      a native dtype via `pkg/backend/compute/convert` before storage.
      The on-device dtype is what `Tensor.DType()` reports.

3.4.3 Async upload path: `UploadAsync(shape, dtype, bytes) (Tensor,
      error)` returns immediately with `StatePending`; the upload
      completes on a dedicated Metal command queue and transitions
      the tensor to `StateReady`. `Tensor.Sync(ctx)` waits the
      command-buffer completion handler.

3.4.4 `*Native` accessors implemented for every Metal-native dtype.
      `BFloat16Native()` on a bf16 buffer returns `[]dtype.BF16`
      aliasing the MTLBuffer's host pointer. Aliasing contract from
      §2.8 enforced via the state machine.

3.4.5 Legacy `UploadFloat64`/`Float64Tensor` path is preserved
      against the old tensor type until Phase 7.

3.4.6 Verification: bf16 weight load from safetensors → Metal upload
      → BFloat16Native read on host → round-trip via Snapshot/Restore
      → BFloat16Native read again → bit-exact equality with the
      original safetensors bytes. Plus throughput benchmark: Metal
      upload of 1 GiB at memcpy speed on the host side and zero
      device-side copies.

Definition of Done for Phase 4: Metal stores bf16/fp16 natively; no
silent narrowing; async upload state machine works under concurrent
load; legacy path still present and tested.

### 3.5 Phase 5: CUDA end-to-end

3.5.1 Host staging buffers use `cudaMallocHost` (page-locked). Device
      storage uses `cudaMalloc`. Upload is an async `cudaMemcpyAsync`
      on a dedicated upload stream.

3.5.2 `Backend.Upload(shape, dtype, bytes)` honors source dtype.
      Native CUDA dtypes depend on compute capability (see §2.13);
      others convert via `pkg/backend/compute/convert` before upload.

3.5.3 Async upload: `UploadAsync` returns immediately, tensor in
      `StatePending`. Kernels on the same stream do not block;
      cross-stream kernels call `Tensor.Sync(ctx)` which awaits the
      CUDA event.

3.5.4 `*Native` accessors materialize a host copy through
      `cudaMemcpyAsync` (D→H) into a tiered-allocator host buffer
      and return its typed slice. The state machine flags the tensor
      `StateMutating` for the duration of the host view.

3.5.5 Legacy `UploadFloat64` path retained.

3.5.6 Verification: same pattern as Metal — bf16 → upload → native
      view → round-trip → bit-exact. Throughput target: PCIe DMA
      bandwidth (≥ 25 GiB/s on PCIe 4 x16) on a 1 GiB upload.

Definition of Done for Phase 5: CUDA stores native dtypes per device
capability; async upload non-blocking; legacy path intact.

### 3.6 Phase 6: XLA end-to-end

3.6.1 Tensor storage in HLO buffers. `Backend.Upload` dispatches
      through the XLA runtime's transfer manager.

3.6.2 `Backend.Upload(shape, dtype, bytes)` honors source dtype.
      Conversions go through `pkg/backend/compute/convert` on the
      host side before transfer.

3.6.3 Async upload via the XLA stream pool. State machine integrated.

3.6.4 `*Native` accessors copy back via the transfer manager.

3.6.5 Legacy `UploadFloat64` path retained.

3.6.6 Verification: bf16 round-trip; throughput at the transfer
      manager's documented ceiling.

Definition of Done for Phase 6: XLA backend symmetric with Metal/CUDA
for dtype-native storage and async upload; legacy path intact.

### 3.7 Phase 7: kill the legacy

Every backend now exposes dtype-native storage through `Backend.Upload`
and the new `Tensor` interface. This phase deletes the float64 lingua
franca.

3.7.1 Sed pass across all 1,161 call sites replacing `Float64Tensor`
      parameter types with `Tensor`. Replace `.CloneFloat64()` with
      `convert.ToFloat64(tensor)` (which dispatches to the SIMD
      conversions from Phase 2).

3.7.2 Update kernel signatures to dispatch on `Tensor.DType()` per
      §2.11. Initially each kernel registers `Float64` as the only
      supported dtype (preserving today's behavior); other dtypes
      land in Phase 8.

3.7.3 Delete `Float64Tensor`, `UploadFloat64`, `DownloadFloat64`,
      `Float64From`, `MustFloat64From`, `MustCloneFloat64`, the old
      `HostTensor` struct with its float64 backing, the 64 MiB
      arena, and every other artifact of the legacy path.

3.7.4 `pkg/runtime/state/tensor.go` rewritten to wrap a
      `tensor.Tensor`. `newTensorFromConfig` takes a `tensor.Backend`
      and calls `backend.Upload`. Snapshot/Restore uses the v1 schema
      from §2.10. `state.Tensor.MigrateTo(backend)` implements the
      `Bind` path the original docstring referenced.

3.7.5 Update `pkg/model/weights/safetensors.go` to call
      `backend.Upload(shape, dtype, rawBytes)` directly without
      upcasting to float64. The "convert everything to float64" path
      is deleted. For dtypes outside the backend's `SupportedDTypes`,
      the loader calls `convert.To<backend-native>` on the host side
      first and uploads the converted bytes.

3.7.6 Verification: full test suite passes. Bf16 model load goes
      end-to-end without ever materializing float64 host storage
      (verified by a custom allocator-tracker assertion that fails if
      any float64 allocation exceeds 1 MiB during the load). Memory
      footprint of the loaded model is bf16-size, not float64-size.

Definition of Done for Phase 7: `Float64Tensor` does not appear in
the codebase; bf16 model load uses bf16 host storage end-to-end;
state.Tensor unified; all existing tests pass.

### 3.8 Phase 8: per-kernel dtype rollouts

This is the long tail. Each kernel family gets a session. Per
AGENTS.md §1 and §5.4 below, **every kernel ships with all five host
ISA variants** — scalar Go reference, AVX-512 (amd64), AVX2 (amd64),
SSE2 (amd64), NEON (arm64) — plus Metal, CUDA, and XLA paths. No
ISA-aliasing, no scalar bodies inside SIMD files, no shared assembly
bodies between ISAs. Each kernel family lands in its own session with
parity tests at `N ∈ {1, 7, 64, 1024, 8192}` against the scalar
reference, tight ULP bounds, and benchmarks pasted.

3.8.1 Math (matmul, add, mul, gelu, softmax, layernorm, rmsnorm): bf16,
      fp16, fp32, fp64 paths with mixed-dtype accumulation. Five ISA
      variants per dtype per op. Parity. Benchmarks.

3.8.2 Attention (flash-attention variants, alibi, rope, sliding window):
      bf16 and fp16 paths. Five ISA variants per dtype per op. Parity.
      Benchmarks.

3.8.3 Convolution: fp32 and bf16 paths on Metal/CUDA's vendor primitives;
      five-ISA host fallback for the scalar reference and CPU-only
      research targets.

3.8.4 Optimizer states (Adam, AdamW, Adamax, Lion, Sophia): fp32 master
      with optional bf16 / fp16 master shadows (parameter shadowing for
      lower-precision training). Document precision invariants.

3.8.5 Quantized inference (int8, int4): full new kernel family. GPTQ,
      AWQ, SmoothQuant loaders. Dequant paths bf16 and fp16. The int4
      packed format from §2.4 is used end-to-end.

3.8.6 FP8 (E4M3, E5M2): bf16 master with fp8 storage. Five ISA
      variants for the host scalar/SIMD fallback (no FP8 SIMD on
      current x86; emulate via fp16 with saturation); Metal and CUDA
      native paths.

Definition of Done for Phase 8: every listed kernel family ships with
five host ISA variants per supported dtype, Metal/CUDA/XLA paths, and
parity + benchmark output pasted. The standard transformer reference
graph runs end-to-end in bf16 on Metal at vendor-primitive throughput.

### 3.9 Phase 9: sparse tensor support

3.9.1 Add `Layout` enum and the `SparseTensor` interface from §2.16
      to `pkg/backend/compute/tensor`.

3.9.2 Implement `HostSparseTensor` for each layout (CSR, CSC, COO,
      BSR) over the tiered allocator. Each layout lives in its own
      file per the file-size rule.

3.9.3 Implement `Backend.UploadSparse` on every backend. Metal/CUDA
      delegate to their respective sparse primitive libraries
      (MPS-Graph sparse on Metal, cuSPARSE on CUDA). Host stores
      values + indices in separate aligned buffers.

3.9.4 Extend the kernel dispatch table to a `(Layout, DType)` two-level
      key. Existing dense kernels register as `(LayoutDense, *)`.

3.9.5 Sparse kernels (CSR / CSC / COO / BSR matmul, sparse attention,
      block-sparse gemm): five-ISA scalar+SIMD fallback plus
      vendor-primitive paths (cuSPARSE, MPS-Graph sparse). Parity
      against dense reference.

3.9.6 Snapshot/Restore (§2.10) extended to carry the sparse index
      payload for non-dense layouts. v1 schema is amended with the
      conditional sparse-index suffix.

3.9.7 `pkg/model/weights/safetensors.go` and any other loader paths
      learn to recognize sparse storage formats from disk (GGUF
      sparse extensions, SafeTensors sparse metadata when present)
      and upload via `UploadSparse`. Dense fallback when the source
      format does not advertise sparsity.

3.9.8 Parity tests at `N ∈ {1, 7, 64, 1024, 8192}` for every sparse
      kernel against the dense reference at the same effective
      sparsity. Benchmarks against dense baseline at sparsity ratios
      `{0.5, 0.9, 0.99}`.

Definition of Done for Phase 9: a 2:4-structured sparse model loads
through `UploadSparse`, runs a forward pass via the sparse matmul
kernel, and produces bit-equivalent results (within bf16 ULP) to the
dense reference; measured speedup at 90% sparsity is ≥ 3× on Metal/CUDA.

### 3.10 Phase 10: distributed / sharded tensors

3.10.1 Add `pkg/backend/compute/distributed` with `ShardingMesh`,
       `ShardingSpec`, `DimSharding`, and `DistributedTensor` per §2.17.

3.10.2 Add `pkg/backend/compute/collective` with `AllReduce`,
       `AllGather`, `ReduceScatter`, `Broadcast` interfaces.
       Implementations: NCCL adapter for CUDA, MPS multi-GPU ring for
       Metal Mac Studio / Mac Pro hardware, pkg/network-transport ring
       for host nodes. Parity against single-device reduction.

3.10.3 Extend the orchestrator's scheduler to plan collective
       placement: each `DistributedTensor` operation lowers to a
       per-shard op-graph plus collective ops at sync points.

3.10.4 Process-group bootstrap via `pkg/network/transport`: rendezvous
       protocol, mesh-rank assignment, peer discovery. Survives
       single-node mocking for testing.

3.10.5 Parity: a tensor `A` sharded `[batch_sharded, model_replicated]`
       across a 4×2 mesh, multiplied with a tensor `B` replicated on
       `batch` and sharded on `model`, must produce a result whose
       gather-to-single-device value matches the unsharded reference
       within bf16 ULP at `N ∈ {1, 7, 64, 1024, 8192}`.

Definition of Done for Phase 10: a 70B model can be loaded as a
2-way pipeline-parallel × 2-way tensor-parallel `DistributedTensor`
across 4 GPUs, runs a forward pass, and matches single-GPU output
within precision bounds.

### 3.11 Phase 11: autograd / tape recording

3.11.1 Add `GradFn`, `Tape`, and the autograd-related `Tensor` methods
       from §2.18.

3.11.2 Every forward kernel in §3.8 is paired with a backward kernel
       that registers a `GradFn` on the tape when any input
       `RequiresGrad()`. Backward kernels go through the same
       five-ISA-variant requirement plus Metal/CUDA/XLA paths.

3.11.3 Gradient accumulation via `Tensor.AccumulateGrad(other Tensor)`.
       `tensor.ZeroGrad(t Tensor) error` resets the gradient to a
       zero tensor of matching shape/dtype.

3.11.4 Mixed-precision autograd: backward of a forward op that
       upcasts (bf16 input → fp32 accumulation → bf16 output)
       follows the same casting rule for `grad_output → grad_input`.
       The kernel documents its precision contract; the autograd
       layer is dtype-blind.

3.11.5 Higher-order gradients (gradient-of-gradient) are deferred to
       a follow-up phase. The tape supports recording derivative ops
       as first-class graph nodes, but second-order requires
       re-recording through the backward kernels and is its own work
       item.

3.11.6 Parity: every backward kernel verified by finite-difference
       check at `N ∈ {1, 7, 64, 1024, 8192}` with relative tolerance
       matched to dtype (1e-6 fp64, 1e-3 fp32, 1e-2 bf16/fp16,
       3e-2 fp8).

Definition of Done for Phase 11: an end-to-end training step on a
small transformer (e.g. nanoGPT-scale) trains for 100 steps with loss
matching a reference PyTorch implementation within bf16 ULP at every
step.

### 3.12 Phase 12: graph-level fusion updates

3.12.1 Update CSE pass in `pkg/backend/compute/orchestrator` to
       compare `(DType, Layout, ShardingSpec)` triples when matching
       expressions.

3.12.2 Update DCE pass to recognize layout/sharding-aware tensor
       lifetimes (sparse indices must not be DCE'd when their
       values tensor is live).

3.12.3 Update the fusion catalog with explicit entries for the new
       fused patterns: `bf16_matmul + bf16_bias + bf16_gelu`,
       `fp16_layernorm + fp16_residual`, `int4_dequant + bf16_matmul`,
       etc. Each entry names source ops, fused op, dtype combination,
       parity bound, target backends.

3.12.4 Forbid fusion across dtype-changing ops unless the entry is in
       the catalog. The orchestrator's pass logs and skips unrecognized
       sequences rather than fusing opportunistically.

3.12.5 Parity tests for every fusion catalog entry at
       `N ∈ {1, 7, 64, 1024, 8192}` against unfused execution.

Definition of Done for Phase 12: every fusion catalog entry has a
parity test; the orchestrator emits zero unrecognized fusions on the
standard transformer reference graph.

---

## 4. Verification requirements

Every phase ships with the verification artifacts required by AGENTS.md §2.

For every kernel that gains a dtype variant: a parity test at
`N ∈ {1, 7, 64, 1024, 8192}` against the scalar Go reference for that
dtype, with tight ULP bounds appropriate to the format (1 ULP for f32, 2
ULP for f16/bf16, 4 ULP for fp8 — these are the round-to-nearest-even
bounds from the relevant standards; if a kernel cannot meet them, the
kernel is wrong, not the test).

For every allocator change: a throughput benchmark at allocation sizes
`{64 B, 1 KiB, 64 KiB, 1 MiB, 64 MiB, 1 GiB}` × `{1, GOMAXPROCS}`
threads, with output pasted in the PR.

For every backend Upload/Download change: a round-trip parity test for
every dtype in `SupportedDTypes`. For bf16/fp16, round-trip is bit-exact.
For dtypes that require conversion at the boundary, the conversion is
documented and parity is asserted against the canonical conversion.

For every interface change in §2: an integration test that exercises
the new contract from end to end (e.g. safetensors load → state.Tensor
→ kernel dispatch → snapshot → restore → kernel dispatch → bit-exact
equality of final activation).

The phrase "for now" is not permitted in any commit message, code
comment, or test name introduced by this work.

---

## 5. Decisions left explicit

These are the choices where multiple reasonable answers exist and a
later contributor might be tempted to revisit. Each is settled here;
revisit only with a documented amendment to this file.

5.1 **Three-tier allocator with no-zero default, no `sync.Pool`.**
Reasoning: performance and correctness. Go's `make` zeros memory at
~10–20 GB/s, so a 140 GB bf16 weight load through `make` burns 10–15
seconds of wall-clock writing zeros that the next instruction
overwrites. `sync.Pool` is not an answer either — the runtime drains
it on every GC cycle, so precisely under the memory pressure that
should make pooling save us, the pool is empty. The tier split
(sharded slab allocator < 1 MiB, mmap + `MADV_DONTNEED` for 1 MiB –
1 GiB, mmap + `MADV_HUGEPAGE` for ≥ 1 GiB) matches the lifetime and
size profile of the three tensor classes (per-token scratch,
per-layer activations, weights/KV cache) without GC interaction at
any tier. The no-zero contract is safe because every `Upload`
overwrites the buffer before any reader; explicit `NewZeroed` covers
the zero-init cases. NUMA awareness on Linux ensures multi-socket
hosts keep tensor data on the node where its kernels run.

5.2 **Native views alias storage; conversion is explicit.** Reasoning:
correctness. Silent conversion makes it impossible to audit precision
losses. Aliasing forces callers to acknowledge mutation semantics.

5.3 **Little-endian everywhere.** Reasoning: correctness. Every wire
format Caramba interoperates with is little-endian; every hardware
target is little-endian. Big-endian appears in `dtype/bfloat16.go` only
as inherited Ollama code with no caller; it is corrected.

5.4 **64-byte host alignment and the AVX-512 mandate.** Reasoning:
performance. 64-byte alignment is the widest SIMD load width we
target. Costs at most 56 bytes of slack per tensor; worth it for
aligned-load throughput on every kernel. The alignment guarantee is
load-bearing: it implies that every SIMD operation must ship an
AVX-512 path in addition to AVX2, SSE2, NEON, and the scalar Go
reference, per AGENTS.md §1. There is no "AVX2 is enough" path —
if a kernel is implemented in SIMD assembly, all five ISA variants
(scalar Go + AVX-512 + AVX2 + SSE2 + NEON) exist as separate kernel
bodies with their own vector instructions.
The full five-variant requirement (scalar + AVX-512 + AVX2 + SSE2 +
NEON) applies to every kernel family in §3.8 and to every conversion
kernel in §3.2 (§2.20).

5.5 **Mixed-dtype kernels are first-class.** Reasoning: correctness and
performance. Standard transformer matmul is bf16 × bf16 with fp32
accumulation; pretending it is "bf16" or "fp32" loses information.
Each kernel declares its accepted combinations.

5.6 **Contiguous storage with offset+length subview only.** Reasoning:
performance and correctness. General strides force every kernel to be
either stride-aware (defeating vectorization via gather instructions
and destroying GPU memory coalescing) or contiguity-demanding (which
silently materializes copies at the framework boundary, making the
"O(1) transpose" promise a lie). Every vendor primitive we hand off
to (cuBLAS, cuDNN, MPS-Graph, oneDNN, XLA HLO) wants contiguous input
and handles transpose via an op-level flag, not via strided views.
Caramba therefore stores tensors strictly contiguous row-major, exposes
a single zero-copy `Slice(start, length)` for offset+length subviews
(safetensors mmap pieces, KV-cache active windows), supports
metadata-only `Reshape` when element order is preserved, and
materializes transpose / permute / broadcast through
`tensor.Contiguous(rearrangement)`. Kernel signatures stay
contiguous-only; the implementation surface stays vectorizable and
coalesced.

5.7 **`Set` is removed.** Reasoning: correctness. In-place shape
mutation breaks every cached invariant and complicates lifetime
reasoning. Callers needing a new shape allocate a new tensor; in-place
math goes through dedicated kernels.

5.8 **Snapshot schema is versioned.** Reasoning: correctness. Future
layout changes (e.g. adding ragged tensors, adding compression) require
a wire-format bump; v1 sets the precedent.

5.9 **`Float64Tensor` is deleted, not deprecated.** Reasoning: it
encodes a wrong assumption into 1,161 call sites. Deprecation leaves
the assumption in place; deletion forces the migration.

5.10 **Metal storage mode is Shared.** Reasoning: performance and
correctness on Apple silicon. Host and GPU share physical memory;
upload is metadata, not copy. The Private storage mode is reserved for
the rare case (large transients) where Apple's GPU benefits from
discrete residency; that path is opt-in per kernel.

5.11 **No `View[T]` wrapper, no `Release()`, no reader counter.**
Reasoning: correctness. The Rust/C++ RAII pattern does not survive
the trip into Go. A `Release()` call that callers must remember
becomes a leak every time a panic or early return skips the defer; a
closure-bearing `View[T]` struct escapes to the heap and produces GC
churn on every native-view call in the hot path. Native accessors
return plain typed slices aliasing storage. Aliasing safety is
enforced by the tensor state machine in §2.8 (StateReady,
StatePending, StateMutating, StateClosed): a host-side view request
on a StatePending tensor errors with `ErrTensorInTransit`, and Close
on a tensor with outstanding native slices errors with
`ErrTensorMutating`. Outstanding-view tracking uses
`runtime.AddCleanup` for best-effort transition back to StateReady;
correctness does not depend on its timing.

5.12 **Per-backend end-to-end migration; legacy path preserved
until kill-the-bridge.** Reasoning: performance and correctness. The
earlier "widen the interface in Phase 2, make storage dtype-native
in Phase 4" sequencing would have routed every tensor op through a
silent double-conversion (bf16 → f64 → bf16 → f64) for the duration
of those two phases. Main would have regressed ~2× under standard
benchmarks and we'd have shipped that regression. The new sequencing
migrates each backend end-to-end (Phases 3–6 in §3) while preserving
the legacy `Float64Tensor` path on every backend until Phase 7's
single sed pass deletes it. Performance never drops below baseline on
main; each backend's new path is fully native from the day it lands.

5.13 **Conversion is a first-class kernel family with five host ISA
variants.** Reasoning: performance and correctness. Throughput-bound
conversions (the inner loop of every weight load and every
dtype-mismatched operation) cannot be left to scalar Go. Per §2.20
and Phase 2, every dtype↔dtype conversion ships scalar + AVX-512 +
AVX2 + SSE2 + NEON variants. FP8↔f32 specifically uses LUT-in-zmm
tricks because there is no native FP8 SIMD on current x86; the
requirement is ≥ 10 GiB/s on the host so weight loads remain
bandwidth-bound rather than conversion-bound.

---

## 6. Scope summary

Everything below is in scope and is specified by this document. None
of these items is deferred or revisited later; they land as the phases
in §3 progress.

NUMA awareness on the host backend lives in §2.7 and §3.3. The host
allocator detects topology at init, binds pages per node, and pins
worker threads to the node owning their tensor data.

Conversion kernels are first-class, specified in §2.20 and
implemented in §3.2. Every dtype↔dtype conversion ships with the full
five-host-ISA variant set (scalar + AVX-512 + AVX2 + SSE2 + NEON) plus
Metal/CUDA/XLA paths.

Sparse tensors are first-class, with `Layout` discrimination in §2.16,
layout-aware kernels in §3.9, and snapshot/restore extended to carry
sparse index payloads in §2.10. Supported layouts: CSR, CSC, COO, BSR.

Distributed and sharded tensors are a first-class abstraction in
§2.17, implemented in §3.10. The `DistributedTensor` interface composes
per-shard `Tensor`s under a `ShardingMesh` + `ShardingSpec`, with
collectives in `pkg/backend/compute/collective` handling cross-shard
communication.

Autograd / tape recording is a first-class feature of the compute
layer, specified in §2.18 and implemented in §3.11. Every forward
kernel in §3.8 ships with a backward kernel; the `Tape` records ops,
`Tensor.Grad()` exposes accumulated gradients, and finite-difference
parity is required on every backward.

Graph-level fusion changes are part of this rewrite. The orchestrator's
CSE, DCE, and fusion passes are made dtype-aware, layout-aware, and
sharding-aware in §2.19 and §3.12. The fusion catalog is explicit; no
opportunistic fusion across dtype changes.

The only items not specified by this document are higher-order
gradients (gradient-of-gradient) in autograd, which §3.11 records as a
follow-up if a use case appears. Everything else is in scope and must
be implemented to the Definition of Done for its phase.

---

## 7. Reading order for implementers

When starting Phase N, read in order: this document (the relevant phase
section plus §2 contract), `AGENTS.md`, the existing code being
replaced, and the test files exercising it. Reason through the kernel
math before writing code. If something in the existing code looks wrong,
read it carefully before concluding it is wrong — there is usually a
load-bearing reason. If you cannot explain the reason, ask before
replacing.

If you reach a point where the doc is silent or ambiguous, stop and
patch the doc. Do not encode a judgement call as a comment in the code
and move on. The doc is the contract; if the contract is incomplete,
fix the contract.
