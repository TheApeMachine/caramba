# Metal Backend Implementation Mandate

This document is an implementation contract for turning `pkg/backend/compute/metal` into a complete, high-performance GPU backend for the platform. The goal is not a smaller backend, a reduced operation set, a guarded path, or an execution mode that declines work. The goal is resident Metal execution for the full compute contract.

Every operation that the platform can lower to compute must have a real Metal path. Every tensor produced inside a Metal graph must remain in Metal storage until an explicit external boundary requests host data. Every optimization that materially affects GPU throughput must be part of the backend, not deferred into an external wrapper.

## Required End State

The backend is complete only when all of the following are true:

1. Every required operation ID has a resident Metal implementation.
2. Every required operation ID has parity tests against the scalar reference.
3. Every required operation ID has a benchmark.
4. Every graph node executes on resident `MTLBuffer` storage.
5. Graph execution performs no intermediate host readback.
6. Parameters, optimizer state, KV cache state, temporaries, and outputs all live in Metal-managed memory.
7. Command submission is batched at graph or subgraph granularity.
8. Memory allocation is planned from graph lifetimes and reused aggressively.
9. Fusion is applied wherever it removes materialization, launch overhead, or memory bandwidth.
10. Performance evidence shows the backend is limited by GPU math, GPU memory bandwidth, or unavoidable graph dependencies, not CPU orchestration.

## Metal Execution Model

Metal graph execution must have exactly one production path:

```text
IR graph
  -> verified typed graph
  -> Metal legality and dtype lowering
  -> Metal graph planner
  -> Metal memory planner
  -> Metal command encoder
  -> resident MTLBuffer outputs
```

`TensorBackend.Apply` is the entry point to keep. It must grow until it covers the complete operation surface.

The host-slice `OperationRegistry` path must be removed from graph execution. Methods that accept `[]float64` and return `[]float64` are not graph kernels. They create host data motion by construction and cannot be part of a high-performance GPU backend.

The executor must make host movement structurally impossible for Metal graph nodes. `DownloadFloat64` belongs only at explicit output, debug, serialization, or test comparison boundaries.

## Runtime Architecture

Implement one Metal runtime object and thread it through the backend:

```text
MetalRuntime
  device
  commandQueue
  commandBufferPool
  pipelineCache
  argumentEncoderCache
  allocator
  uploadRing
  readbackRing
  eventPool
  profiler
  captureController
  diagnosticSink
```

All operation families must use this runtime:

- activation,
- math,
- attention,
- masking,
- shape,
- embedding,
- projection,
- convolution,
- pooling,
- optimizer,
- training,
- Hawkes,
- VSA,
- active inference,
- predictive coding,
- Markov blanket,
- causal,
- memory,
- tokenizer and data kernels that execute inside compute graphs.

The runtime owns device selection, command queue lifetime, pipeline lifetime, memory pools, event timing, capture labels, and error reporting. Per-file global devices and queues must be eliminated.

## Command Submission

Every current per-kernel synchronous submission must be replaced with batched encoding.

Required behavior:

- encode all ready nodes for a graph segment into command buffers,
- use command-buffer ordering and Metal events to preserve dependencies,
- keep multiple command buffers in flight,
- wait only at explicit graph output or readback boundaries,
- label every command buffer, encoder, and kernel dispatch with graph ID, node ID, and operation ID,
- collect GPU timestamps for benchmarks,
- surface command-buffer failures with the exact node and kernel label.

Elementwise chains must not become one command buffer wait per operation. They must be fused or encoded as a batch.

## Memory Management

Replace per-operation output allocation with a Metal allocator driven by graph lifetimes.

The allocator must support:

- persistent parameter buffers,
- persistent optimizer-state buffers,
- persistent KV-cache buffers,
- graph output buffers,
- temporary scratch buffers,
- reusable size-class pools,
- aliasing views,
- explicit staging upload buffers,
- explicit readback buffers,
- peak-memory and reuse accounting.

`newMetalTensor` should become an allocator call. The allocator receives dtype, shape, layout, lifetime class, and aliasing constraints. It returns a tensor handle backed by an `MTLBuffer` or by metadata that aliases an existing buffer.

Views and reshapes must be metadata operations when layout permits. Transpose, gather, scatter, and layout conversion must be resident kernels when physical movement is required.

## Tensor Model

The current `Float64Tensor` surface is not sufficient as the long-term resident GPU tensor interface. Metal tensors must carry:

- dtype,
- logical shape,
- physical shape,
- strides,
- storage offset,
- layout,
- byte length,
- storage mode,
- aliasing parent,
- lifetime class,
- producing node ID,
- owning runtime.

The Metal backend must support explicit dtype execution. `float32`, `float16`, and future packed formats are separate capabilities with their own parity references. A `float32` Metal kernel must not be represented as satisfying a `float64` contract. A `float64` Metal contract requires a real device implementation of that contract.

## Operation Coverage

The Metal backend must implement the complete operation surface as resident kernels or resident calls to measured Apple GPU primitives.

### Core Math

Implement resident coverage for:

- add, sub, mul, div,
- neg, abs, sign,
- sqrt, rsqrt, exp, log,
- clamp, min, max,
- scalar variants,
- broadcasting variants,
- softmax,
- logsumexp,
- reductions,
- dropout,
- loss helpers,
- normalization helpers.

All operations must handle the contract sizes `N = 1, 7, 64, 1024, 8192` plus shape-specific matrix and tensor cases.

### Activations

Implement resident coverage for:

- ReLU,
- LeakyReLU,
- GELU,
- Tanh,
- Sigmoid,
- Swish,
- SELU,
- SwiGLU,
- every additional activation in the required operation table.

The math definition must match the scalar reference for the dtype contract.

### Matrix Multiplication And Projection

Implement a full GEMM and projection strategy:

- rank-2 matmul,
- batched matmul,
- vector-matrix decode kernels,
- projection.linear,
- fused QKV projection,
- tied embedding projection,
- matmul + bias,
- matmul + bias + activation,
- matmul + residual,
- shape-specialized kernels,
- `simdgroup_matrix` kernels on supported GPU families,
- MPS and MPSGraph measured baselines,
- automatic measured kernel selection per shape class.

Weights must be materialized as persistent resident tensors before execution. The graph executor must not upload weights from node metadata during execution.

### Attention

Implement complete resident attention:

- SDPA,
- MQA,
- GQA,
- sliding-window attention,
- causal masking,
- arbitrary mask application,
- mask generation,
- RoPE,
- ALiBi,
- RoPE + Q/K preparation fusion,
- QK + scale + mask + softmax + V fusion,
- prefill-specialized kernels,
- decode-specialized kernels,
- KV append,
- KV repack,
- KV cache capacity growth,
- GQA without KV head expansion.

Attention must use online softmax accumulation where it reduces memory traffic. KV cache state must stay in Metal buffers across decode steps.

### Shape And Layout

Implement resident coverage for:

- reshape,
- transpose,
- concat,
- split,
- slice,
- gather,
- scatter,
- view_as_heads,
- merge_heads,
- last_token,
- upsample_nearest2d,
- layout conversion.

Metadata-only shape operations must not allocate. Movement operations must be explicit resident kernels.

### Embedding

Implement resident coverage for:

- token embedding,
- positional lookup where present,
- tied embedding projection,
- embedding gradients required by training.

Embedding weights must be persistent resident tensors.

### Convolution And Pooling

Implement resident coverage for:

- Conv1D,
- Conv2D,
- Conv3D,
- ConvTranspose2D,
- grouped convolution,
- depthwise convolution,
- MaxPool2D,
- AvgPool2D,
- AdaptiveAvgPool2D,
- AdaptiveMaxPool2D.

Kernels must support the platform's layout contract and record any additional optimized layouts in tensor metadata.

### Optimizers And Training

Implement resident optimizer execution:

- Adam,
- AdamW,
- AdaMax,
- SGD,
- Lion,
- RMSProp,
- Hebbian,
- LARS,
- LAMB,
- AdaGrad,
- AdaDelta,
- L-BFGS.

Optimizer state must be resident. Multi-tensor updates must be fused. Gradient clipping, norm computation, loss reduction, cross entropy, MSE, accuracy, perplexity, and F1 must execute without intermediate readback.

### Research Operation Families

Implement resident coverage for:

- Hawkes intensity,
- Hawkes kernel matrix,
- Hawkes log likelihood,
- Hawkes simulation,
- VSA bind,
- VSA bundle,
- VSA similarity,
- VSA permute,
- VSA inverse permute,
- active inference belief update,
- active inference expected free energy,
- active inference free energy,
- active inference precision weighting,
- predictive coding prediction,
- predictive coding prediction error,
- predictive coding representation update,
- predictive coding weight update,
- Markov blanket flow operations,
- Markov blanket mutual information,
- Markov blanket partition,
- causal backdoor adjustment,
- causal frontdoor adjustment,
- causal CATE,
- causal counterfactual,
- causal DAG Markov factorization,
- causal do-calculus,
- causal IV estimate.

Every operation family must have resident tensor methods, graph dispatch wiring, parity tests, and benchmarks.

## Fusion Requirements

Fusion is mandatory for a high-performance Metal backend.

Implement fusions for:

- elementwise chains,
- matmul + bias,
- matmul + bias + activation,
- matmul + residual,
- matmul + bias + activation + residual,
- normalization + scale + bias,
- RoPE + attention preparation,
- mask + softmax,
- QK + mask + softmax + V,
- projection + activation,
- loss + reduction,
- optimizer multi-tensor updates.

The compiler must select fused Metal operations when they reduce launches, memory traffic, allocations, or read/write passes. Fusion legality must come from the Metal operation table.

## Operation Table

Create a single Metal operation table. It is the source of truth for implementation status.

Each entry must include:

```text
operation_id
dtype
shape_constraints
layout_constraints
resident_kernel_symbol
fused_variants
runtime_entrypoint
parity_test
benchmark
gpu_family_requirements
```

The table drives:

- capability registration,
- lowering legality,
- graph dispatch,
- build-time symbol validation,
- parity test generation,
- benchmark generation,
- documentation.

The final table must contain every required operation ID.

## Compiler Integration

The compiler must lower to Metal with concrete execution facts:

- dtype,
- layout,
- aliasing,
- lifetime,
- fusion group,
- memory class,
- kernel variant,
- expected output shape,
- dependency list.

Lowering should produce a Metal execution plan, not a list of generic node calls. The execution plan is what the runtime encodes into command buffers.

## Build Integration

Metal shader artifacts must be generated and validated by the build:

- compile every `.metal` source into `.metallib`,
- validate every declared kernel symbol exists,
- validate every operation table entry has a runtime entrypoint,
- validate every runtime entrypoint has tests and benchmarks,
- fail build when coverage is incomplete.

This makes missing Metal work impossible to hide.

## Testing Contract

Every operation needs:

- scalar reference parity at `N = 1, 7, 64, 1024, 8192`,
- shape-specific parity for rank-sensitive operations,
- dtype-specific parity,
- resident-location assertions for every input and output,
- graph-level tests that detect any intermediate `DownloadFloat64`,
- graph-level tests for persistent parameters,
- graph-level tests for persistent optimizer state,
- graph-level tests for KV cache residency,
- benchmarks for resident execution,
- benchmarks for upload and readback boundaries,
- end-to-end graph benchmarks.

Tolerance must be defined by dtype and ULP policy, not by arbitrary broad epsilon.

## Benchmark Contract

Every benchmark must report:

- GPU elapsed time,
- CPU wall time,
- command-buffer count,
- kernel dispatch count,
- allocation count,
- allocated bytes,
- reused bytes,
- transfer bytes,
- effective bandwidth,
- effective FLOP/s where meaningful,
- peak resident memory.

Benchmark groups:

- microkernels,
- fused kernels,
- attention prefill,
- attention decode,
- convolution blocks,
- optimizer steps,
- research kernels,
- complete graph execution.

The benchmark suite must compare:

- scalar CPU,
- SIMD CPU,
- current Metal kernel,
- fused Metal kernel,
- MPS or MPSGraph baseline where applicable.

## Implementation Sequence

1. Build `MetalRuntime`.
2. Move device, queue, pipeline, and capture ownership into `MetalRuntime`.
3. Build the Metal operation table.
4. Wire capabilities, lowering, tests, and benchmarks to the operation table.
5. Build the resident allocator.
6. Add lifetime-based memory planning.
7. Replace per-operation output allocation with allocator requests.
8. Replace synchronous per-kernel waits with command-buffer batch encoding.
9. Add graph-level no-readback instrumentation.
10. Materialize parameters as persistent resident tensors.
11. Materialize optimizer state as persistent resident tensors.
12. Expand `TensorBackend.Apply` to every required operation ID.
13. Add missing resident kernels by operation family.
14. Add required fusions.
15. Add MPS/MPSGraph measured baselines.
16. Add GPU timing and memory accounting.
17. Run the complete parity suite.
18. Run the complete benchmark suite.
19. Publish benchmark output with the backend change.

## Completion Gate

The Metal backend is accepted when:

- the operation table covers every required operation ID,
- build validation proves every operation has a resident symbol and entrypoint,
- graph tests prove no intermediate host readback,
- parity tests pass for every operation and dtype contract,
- benchmarks exist and run for every operation family,
- end-to-end graph benchmarks show command-buffer batching,
- allocator metrics show buffer reuse,
- performance data shows GPU execution dominates CPU orchestration,
- all results are pasted with the change claiming completion.

