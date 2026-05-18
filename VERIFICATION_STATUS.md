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

### 2026-05-18 Metal vision kernel expansion

This adds real Metal device kernels across `float32`, `float16`, and
`bfloat16` storage for:

- `conv2d_{float32,float16,bfloat16}`
- `conv1d_{float32,float16,bfloat16}`
- `conv3d_{float32,float16,bfloat16}`
- `conv_transpose2d_{float32,float16,bfloat16}`
- `max_pool2d_{float32,float16,bfloat16}`
- `avg_pool2d_{float32,float16,bfloat16}`
- `adaptive_avg_pool2d_{float32,float16,bfloat16}`
- `adaptive_max_pool2d_{float32,float16,bfloat16}`

The public registry entries are `conv1d`, `conv2d`, `conv3d`,
`conv_transpose2d`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`,
and `adaptive_max_pool2d` for all three storage dtypes. The shaders
live in `pkg/backend/device/metal/vision.metal` and run through
`pkg/backend/device/metal/bridge_vision_*.m`.

The parity cases use `N ∈ {1, 7, 64, 1024, 8192}` as output spatial
width for `conv1d`, `conv2d`, and `conv3d`, and input width for
`conv_transpose2d`. The convolution kernels use the scalar registry's
default stride 1 / padding 0 / dilation 1 semantics. Standard pooling
uses the scalar registry's 2x2 stride-2 window; adaptive pooling uses
output-driven integer regions. The expected values are computed from
dtype-stored inputs and checked with tight ULP bounds on the stored
output representation. `conv_transpose2d` is implemented as a
per-output gather kernel so no two GPU threads write the same output
element.

The Metal dense registry now has 195 verified signatures: 102
elementwise, 27 shape, 6 matmul, 3 softmax, 6 normalization, 12
projection/model, 15 transformer embedding/masking, and 24 vision
signatures.

Focused vision parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_Metal(Convolution|Vision)DTypes' -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.865s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	1.996s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	2.224s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.471s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	0.858s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	2.641s
```

Metal vision benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunVisionDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunVisionDTypes/f32/conv1d-16                       	   11176	    106517 ns/op	  385.63 MB/s	    1417 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/conv2d-16                       	   10000	    103376 ns/op	  676.33 MB/s	    1440 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/conv3d-16                       	   10000	    105382 ns/op	  508.55 MB/s	    1488 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/conv_transpose2d-16             	   12168	     98427 ns/op	  752.60 MB/s	    1440 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/max_pool2d-16                   	   12120	     98932 ns/op	 2484.13 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/avg_pool2d-16                   	   10000	    103610 ns/op	 2371.96 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/adaptive_avg_pool2d-16          	   10000	    101404 ns/op	 1294.93 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f32/adaptive_max_pool2d-16          	   10000	    102521 ns/op	 1280.83 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/conv1d-16                       	   12238	     98387 ns/op	  208.75 MB/s	    1416 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/conv2d-16                       	   12031	     99569 ns/op	  351.09 MB/s	    1440 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/conv3d-16                       	   10000	    377695 ns/op	   70.95 MB/s	    1488 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/conv_transpose2d-16             	    1819	    693823 ns/op	   53.38 MB/s	    1440 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/max_pool2d-16                   	    1660	    691781 ns/op	  177.63 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/avg_pool2d-16                   	    2036	    650273 ns/op	  188.97 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/adaptive_avg_pool2d-16          	    1880	    691122 ns/op	   95.00 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/f16/adaptive_max_pool2d-16          	    1627	    715274 ns/op	   91.79 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/conv1d-16                      	    1778	    650861 ns/op	   31.56 MB/s	    1416 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/conv2d-16                      	    1803	    556548 ns/op	   62.81 MB/s	    1440 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/conv3d-16                      	    5856	    197884 ns/op	  135.41 MB/s	    1488 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/conv_transpose2d-16            	    7365	    142829 ns/op	  259.32 MB/s	    1440 B/op	       9 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/max_pool2d-16                  	   10000	    101981 ns/op	 1204.93 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/avg_pool2d-16                  	   12028	     99777 ns/op	 1231.55 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/adaptive_avg_pool2d-16         	   12176	     98928 ns/op	  663.67 MB/s	    1352 B/op	       6 allocs/op
BenchmarkKernel_RunVisionDTypes/bf16/adaptive_max_pool2d-16         	   12045	    100288 ns/op	  654.68 MB/s	    1352 B/op	       6 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	30.462s
```

### 2026-05-18 Metal transformer embedding and masking expansion

This adds real Metal device kernels across `float32`, `float16`, and
`bfloat16` storage for:

- `embedding_lookup_{float32,float16,bfloat16}`
- `embedding_bag_{float32,float16,bfloat16}`
- `apply_mask_{float32,float16,bfloat16}`
- `causal_mask_{float32,float16,bfloat16}`
- `alibi_bias_{float32,float16,bfloat16}`

The public registry entries are `embedding_lookup`, `embedding_bag`,
`apply_mask`, `causal_mask`, and `alibi_bias` for all three storage
dtypes. The shaders live in
`pkg/backend/device/metal/transformer.metal` and run through the
embedding and masking bridge files under
`pkg/backend/device/metal/bridge_transformer_*.m`.

Embedding kernels use a device-visible validation buffer. Invalid
token ids, invalid bag offsets, or out-of-range bag spans set that
buffer from the shader; the Objective-C completion handler reads it
after command completion and reports the error through the existing
Metal completion registry. That prevents out-of-bounds table reads
without moving embedding math to the host.

The parity cases use `N ∈ {1, 7, 64, 1024, 8192}` as hidden width for
embedding ops, element count for `apply_mask`, and key width for
`causal_mask` / `alibi_bias`. Float32 copy/add/mask parity is bitwise
where the operation is exact; float16 and bfloat16 parity use 0- or
1-ULP bounds on the stored 16-bit representation depending on whether
the operation accumulates.

RoPE is not promoted in this pass. A direct Metal shader using
`precise::sin/cos` produced a correct rotation shape but missed a
defensible ULP contract against the scalar double-reference at small
cancellation outputs. It needs a different accuracy strategy before it
can be marked verified.

The Metal dense registry now has 171 verified signatures: 102
elementwise, 27 shape, 6 matmul, 3 softmax, 6 normalization, 12
projection/model, and 15 transformer embedding/masking signatures.

Focused transformer parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_Metal(Embedding|MaskingAndPositional)DTypes' -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.602s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	1.029s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	1.137s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.335s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	0.833s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.510s
```

Metal transformer benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunTransformerDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunTransformerDTypes/f32/embedding_lookup-16         	   10669	    113925 ns/op	 9204.12 MB/s	    1337 B/op	       7 allocs/op
BenchmarkKernel_RunTransformerDTypes/f32/embedding_bag-16            	   11904	    246126 ns/op	 4792.86 MB/s	    1352 B/op	       8 allocs/op
BenchmarkKernel_RunTransformerDTypes/f32/apply_mask-16               	    7738	    139846 ns/op	22494.23 MB/s	    1296 B/op	       4 allocs/op
BenchmarkKernel_RunTransformerDTypes/f32/causal_mask-16              	   10000	    116249 ns/op	 9020.09 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunTransformerDTypes/f32/alibi_bias-16               	   10000	    114753 ns/op	18275.33 MB/s	    1312 B/op	       5 allocs/op
BenchmarkKernel_RunTransformerDTypes/f16/embedding_lookup-16         	    8182	    343721 ns/op	 1525.33 MB/s	    1336 B/op	       7 allocs/op
BenchmarkKernel_RunTransformerDTypes/f16/embedding_bag-16            	    5300	    201978 ns/op	 2920.24 MB/s	    1352 B/op	       8 allocs/op
BenchmarkKernel_RunTransformerDTypes/f16/apply_mask-16               	    5205	    226901 ns/op	 6931.94 MB/s	    1296 B/op	       4 allocs/op
BenchmarkKernel_RunTransformerDTypes/f16/causal_mask-16              	    4938	    248547 ns/op	 2109.41 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunTransformerDTypes/f16/alibi_bias-16               	    8911	    127825 ns/op	 8203.22 MB/s	    1312 B/op	       5 allocs/op
BenchmarkKernel_RunTransformerDTypes/bf16/embedding_lookup-16        	   10000	    114930 ns/op	 4561.80 MB/s	    1336 B/op	       7 allocs/op
BenchmarkKernel_RunTransformerDTypes/bf16/embedding_bag-16           	   11038	    109685 ns/op	 5377.42 MB/s	    1352 B/op	       8 allocs/op
BenchmarkKernel_RunTransformerDTypes/bf16/apply_mask-16              	   10000	    111599 ns/op	14093.92 MB/s	    1296 B/op	       4 allocs/op
BenchmarkKernel_RunTransformerDTypes/bf16/causal_mask-16             	   10000	    114130 ns/op	 4593.77 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunTransformerDTypes/bf16/alibi_bias-16              	    9412	    116387 ns/op	 9009.36 MB/s	    1312 B/op	       5 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	21.290s
```

### 2026-05-18 Metal projection and LoRA kernel expansion

This adds real Metal device kernels for projection and model-adapter
ops across `float32`, `float16`, and `bfloat16` storage:

- `linear_{float32,float16,bfloat16}`
- `fused_qkv_{float32,float16,bfloat16}`
- `lora_merge_{float32,float16,bfloat16}`
- `lora_apply_stage1_{float32,float16,bfloat16}`
- `lora_apply_stage2_{float32,float16,bfloat16}`

The public registry entries are `linear`, `fused_qkv`, `lora_merge`,
and `lora_apply` for all three storage dtypes. The shaders live in
`pkg/backend/device/metal/projection.metal` and run through
`pkg/backend/device/metal/bridge_projection_darwin.m`.

`linear` and `fused_qkv` use 16x16 tiled threadgroups with shared
input and weight tiles. `fused_qkv` computes query/key/value in one
pass over the input tile. `lora_merge` computes
`baseWeight + A @ B` per output element. `lora_apply` allocates a
device-resident float32 scratch tensor, computes `B @ input` once in
stage 1, then applies `A @ scratch` plus `baseOut` in stage 2. Both
LoRA stages are encoded into one Metal command buffer and tracked by
one completion token, so scratch lifetime is protected by the same
use-counting path as user-visible tensors.

The parity cases use `N ∈ {1, 7, 64, 1024, 8192}` as the projection
inner dimension. Float32 parity is verified within 1 ULP. Float16 and
bfloat16 parity are verified within 1 ULP on the stored 16-bit
representation.

The Metal dense registry now has 156 verified signatures: 102
elementwise, 27 shape, 6 matmul, 3 softmax, 6 normalization, and 12
projection/model signatures.

Focused projection parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_MetalProjectionDTypes' -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.819s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	1.139s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.313s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.918s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	1.350s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.693s
```

Metal projection benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunProjectionDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunProjectionDTypes/f32/linear-16         	    8210	    144876 ns/op	 6340.11 MB/s	    1393 B/op	       9 allocs/op
BenchmarkKernel_RunProjectionDTypes/f32/fused_qkv-16     	    7682	    152268 ns/op	12501.71 MB/s	    1680 B/op	      12 allocs/op
BenchmarkKernel_RunProjectionDTypes/f32/lora_merge-16    	   10000	    115241 ns/op	18482.24 MB/s	    1400 B/op	       9 allocs/op
BenchmarkKernel_RunProjectionDTypes/f32/lora_apply-16    	    7146	    166260 ns/op	 2562.16 MB/s	    1824 B/op	      14 allocs/op
BenchmarkKernel_RunProjectionDTypes/f16/linear-16        	    8036	    150810 ns/op	 3045.32 MB/s	    1392 B/op	       9 allocs/op
BenchmarkKernel_RunProjectionDTypes/f16/fused_qkv-16     	    7188	    159047 ns/op	 5984.44 MB/s	    1680 B/op	      12 allocs/op
BenchmarkKernel_RunProjectionDTypes/f16/lora_merge-16    	   10000	    122497 ns/op	 8693.77 MB/s	    1400 B/op	       9 allocs/op
BenchmarkKernel_RunProjectionDTypes/f16/lora_apply-16    	    7149	    160228 ns/op	 1329.31 MB/s	    1824 B/op	      14 allocs/op
BenchmarkKernel_RunProjectionDTypes/bf16/linear-16       	    7852	    146818 ns/op	 3128.12 MB/s	    1392 B/op	       9 allocs/op
BenchmarkKernel_RunProjectionDTypes/bf16/fused_qkv-16    	    7472	    159344 ns/op	 5973.28 MB/s	    1680 B/op	      12 allocs/op
BenchmarkKernel_RunProjectionDTypes/bf16/lora_merge-16   	    9260	    121218 ns/op	 8785.48 MB/s	    1400 B/op	       9 allocs/op
BenchmarkKernel_RunProjectionDTypes/bf16/lora_apply-16   	    7287	    161392 ns/op	 1319.72 MB/s	    1824 B/op	      14 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	14.580s
```

### 2026-05-18 Metal extended unary elementwise expansion

This adds real Metal device kernels for 15 dense unary math and
activation ops:

- `rsqrt_{float32,float16,bfloat16}`
- `exp_{float32,float16,bfloat16}`
- `log_{float32,float16,bfloat16}`
- `sin_{float32,float16,bfloat16}`
- `cos_{float32,float16,bfloat16}`
- `tanh_{float32,float16,bfloat16}`
- `sigmoid_{float32,float16,bfloat16}`
- `silu_{float32,float16,bfloat16}`
- `swish_{float32,float16,bfloat16}`
- `softsign_{float32,float16,bfloat16}`
- `elu_{float32,float16,bfloat16}`
- `selu_{float32,float16,bfloat16}`
- `leaky_relu_{float32,float16,bfloat16}`
- `hardsigmoid_{float32,float16,bfloat16}`
- `hardswish_{float32,float16,bfloat16}`

The kernels live in `pkg/backend/device/metal/elementwise_extended.metal`
and run through the existing `runMetalUnaryElementwise` path in
`pkg/backend/device/metal/bridge_elementwise_darwin.m`. Float16 and
bfloat16 read native storage, compute the scalar math in float32, and
write native storage. The parity cases use `N ∈ {1, 7, 64, 1024, 8192}`.
F32 parity uses tight per-op ULP bounds: 1 ULP for `leaky_relu`, 2 ULP
for `rsqrt`, `softsign`, and `hardswish`, 8 ULP for transcendental
ops and exp-derived activations, and 16 ULP for `hardsigmoid` because
Metal division by 6 differs from the scalar reference by up to 13 ULP
near zero. Float16 and bfloat16 parity use 1- or 2-ULP bounds on the
stored 16-bit representation.

Exact `gelu` is not promoted in this pass because this Metal SDK does
not expose `erf`; substituting the tanh variant under the exact name
would violate the scalar definition. `softplus`, `mish`, and
`gelu_tanh` were also not promoted because they did not satisfy tight
f32 ULP parity against the scalar reference on this SDK.

The Metal dense registry now has 144 verified signatures: 102
elementwise, 27 shape, 6 matmul, 3 softmax, and 6 normalization
signatures.

Focused extended unary parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_MetalExtendedUnaryElementwiseDTypes' -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.659s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	1.463s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.413s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	1.322s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	0.765s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.856s
```

Metal extended unary benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunExtendedUnaryElementwiseDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/rsqrt-16   	    9772	    122316 ns/op	 535.79 MB/s	    1289 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/exp-16     	   10000	    120859 ns/op	 542.25 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/log-16     	   10000	    118758 ns/op	 551.84 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/sin-16     	   10000	    118009 ns/op	 555.35 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/cos-16     	   10000	    115248 ns/op	 568.65 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/tanh-16    	   10000	    117118 ns/op	 559.57 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/sigmoid-16 	    9085	    116854 ns/op	 560.83 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/silu-16    	   10000	    115047 ns/op	 569.64 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/swish-16   	   10000	    114661 ns/op	 571.56 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/softsign-16         	    9907	    121251 ns/op	 540.50 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/elu-16              	   10000	    114845 ns/op	 570.65 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/selu-16             	    9097	    114667 ns/op	 571.54 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/leaky_relu-16       	   10000	    116130 ns/op	 564.33 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/hardsigmoid-16      	    9570	    118546 ns/op	 552.83 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f32/hardswish-16        	    9858	    115211 ns/op	 568.83 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/rsqrt-16            	   10000	    115811 ns/op	 282.94 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/exp-16              	    9902	    118710 ns/op	 276.03 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/log-16              	    9864	    115091 ns/op	 284.71 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/sin-16              	    9590	    116125 ns/op	 282.18 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/cos-16              	    9398	    118587 ns/op	 276.32 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/tanh-16             	   10000	    118979 ns/op	 275.41 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/sigmoid-16          	   10000	    115364 ns/op	 284.04 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/silu-16             	   10000	    115386 ns/op	 283.98 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/swish-16            	   10000	    116048 ns/op	 282.37 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/softsign-16         	    9723	    123620 ns/op	 265.07 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/elu-16              	   10000	    112232 ns/op	 291.97 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/selu-16             	    9154	    115536 ns/op	 283.62 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/leaky_relu-16       	    9890	    122013 ns/op	 268.56 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/hardsigmoid-16      	   10000	    114178 ns/op	 286.99 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/f16/hardswish-16        	    9246	    124229 ns/op	 263.77 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/rsqrt-16           	    9831	    119290 ns/op	 274.69 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/exp-16             	    9956	    118298 ns/op	 277.00 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/log-16             	    9272	    116900 ns/op	 280.31 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/sin-16             	    9378	    115353 ns/op	 284.07 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/cos-16             	    9987	    116221 ns/op	 281.95 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/tanh-16            	   10000	    116197 ns/op	 282.00 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/sigmoid-16         	   10000	    121997 ns/op	 268.60 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/silu-16            	   10000	    120717 ns/op	 271.44 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/swish-16           	   10000	    114777 ns/op	 285.49 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/softsign-16        	    9386	    113516 ns/op	 288.66 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/elu-16             	    9664	    125110 ns/op	 261.91 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/selu-16            	    9500	    116912 ns/op	 280.28 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/leaky_relu-16      	   10000	    113568 ns/op	 288.53 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/hardsigmoid-16     	   10000	    114555 ns/op	 286.05 MB/s	    1288 B/op	       4 allocs/op
BenchmarkKernel_RunExtendedUnaryElementwiseDTypes/bf16/hardswish-16       	   10000	    119183 ns/op	 274.94 MB/s	    1288 B/op	       4 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	53.361s
```

### 2026-05-18 Metal normalization kernel expansion

This adds real Metal device kernels for last-dimension normalization:

- `layernorm_{float32,float16,bfloat16}`
- `rmsnorm_{float32,float16,bfloat16}`

The kernels live in `pkg/backend/device/metal/normalization.metal` and
run through `pkg/backend/device/metal/bridge_normalization_darwin.m`.
Each row is assigned one 256-thread threadgroup. LayerNorm performs
parallel mean and variance reductions, then writes
`(x - mean) / sqrt(variance + eps) * scale + bias` in the target storage
dtype. RMSNorm performs a parallel mean-square reduction, then writes
`x / sqrt(mean_square + eps) * scale` in the target storage dtype.
Float16 and bfloat16 read native storage, compute in float32, and write
native storage. The parity cases use rows=13 and
`N ∈ {1, 7, 64, 1024, 8192}` as the normalized width. F32 parity uses
a 32-ULP bound against the scalar reference because the device path
uses parallel reduction order; f16 and bf16 parity use a 2-ULP bound on
the stored 16-bit representation.

The Metal dense registry now has 99 verified signatures: 57
elementwise, 27 shape, 6 matmul, 3 softmax, and 6 normalization
signatures.

Focused host normalization and NEON AXPY parity command:

```
go test ./pkg/backend/compute/kernels -run 'Test(LayerNorm|RMSNorm)(Float32|Float16AndBFloat16)?$|TestAxpyFloat32NEONAsmParity' -count=1 -v
=== RUN   TestAxpyFloat32NEONAsmParity
=== RUN   TestAxpyFloat32NEONAsmParity/N=1
=== RUN   TestAxpyFloat32NEONAsmParity/N=7
=== RUN   TestAxpyFloat32NEONAsmParity/N=64
=== RUN   TestAxpyFloat32NEONAsmParity/N=1024
=== RUN   TestAxpyFloat32NEONAsmParity/N=8192
--- PASS: TestAxpyFloat32NEONAsmParity (0.00s)
=== RUN   TestLayerNormFloat16AndBFloat16
--- PASS: TestLayerNormFloat16AndBFloat16 (0.00s)
=== RUN   TestRMSNormFloat16AndBFloat16
--- PASS: TestRMSNormFloat16AndBFloat16 (0.00s)
=== RUN   TestRMSNormFloat32
--- PASS: TestRMSNormFloat32 (0.00s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	0.500s
```

Focused Metal matmul + normalization + softmax parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_Metal(LayerNorm|RMSNorm|Softmax|MatMul)' -count=1 -v
=== RUN   TestKernelRegistry_MetalMatMulDTypes
--- PASS: TestKernelRegistry_MetalMatMulDTypes (0.06s)
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes
--- PASS: TestKernelRegistry_MetalMatMulAddDTypes (0.02s)
=== RUN   TestKernelRegistry_MetalLayerNormDTypes
--- PASS: TestKernelRegistry_MetalLayerNormDTypes (0.01s)
=== RUN   TestKernelRegistry_MetalRMSNormDTypes
--- PASS: TestKernelRegistry_MetalRMSNormDTypes (0.01s)
=== RUN   TestKernelRegistry_MetalSoftmaxDTypes
--- PASS: TestKernelRegistry_MetalSoftmaxDTypes (0.01s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.667s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	1.916s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.374s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.546s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	0.878s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.237s
```

Metal matmul + normalization + softmax benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_Run(Normalization|Softmax|MatMul)DTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunMatMulDTypes/f32/matmul-16         	    9292	    127125 ns/op	1546.57 MB/s	    1345 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/f32/matmul_add-16     	    8172	    137358 ns/op	1435.08 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunMatMulDTypes/f16/matmul-16         	    8396	    132214 ns/op	 743.52 MB/s	    1344 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/f16/matmul_add-16     	    7958	    132621 ns/op	 743.17 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunMatMulDTypes/bf16/matmul-16        	    9738	    121018 ns/op	 812.31 MB/s	    1344 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/bf16/matmul_add-16    	    7452	    140799 ns/op	 700.00 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunNormalizationDTypes/f32/layernorm-16         	    9093	    134746 ns/op	7842.66 MB/s	    1336 B/op	       7 allocs/op
BenchmarkKernel_RunNormalizationDTypes/f32/rmsnorm-16           	    9661	    117165 ns/op	8984.49 MB/s	    1320 B/op	       6 allocs/op
BenchmarkKernel_RunNormalizationDTypes/f16/layernorm-16         	    9680	    124281 ns/op	4251.53 MB/s	    1336 B/op	       7 allocs/op
BenchmarkKernel_RunNormalizationDTypes/f16/rmsnorm-16           	   10000	    100555 ns/op	5234.30 MB/s	    1320 B/op	       6 allocs/op
BenchmarkKernel_RunNormalizationDTypes/bf16/layernorm-16        	   10000	    102621 ns/op	5148.87 MB/s	    1336 B/op	       7 allocs/op
BenchmarkKernel_RunNormalizationDTypes/bf16/rmsnorm-16          	   10000	    107127 ns/op	4913.18 MB/s	    1320 B/op	       6 allocs/op
BenchmarkKernel_RunSoftmaxDTypes/f32-16                         	   11516	    103555 ns/op	10125.81 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunSoftmaxDTypes/f16-16                         	   10000	    103044 ns/op	5087.98 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunSoftmaxDTypes/bf16-16                        	   10000	    108715 ns/op	4822.61 MB/s	    1304 B/op	       5 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	17.039s
```

NEON AXPY benchmark output:

```
go test ./pkg/backend/compute/kernels -run '^$' -bench 'BenchmarkAxpyFloat32' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/compute/kernels
cpu: Apple M4 Max
BenchmarkAxpyFloat32NEONAsm/N=64-16         	189399170	         6.317 ns/op	121573.47 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32NEONAsm/N=1024-16       	21812976	        57.21 ns/op	214801.24 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32NEONAsm/N=8192-16       	 2805793	       430.9 ns/op	228158.97 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32NEONAsm/N=65536-16      	  200014	      6045 ns/op	130093.47 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32Scalar/N=64-16          	48115878	        23.54 ns/op	32625.66 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32Scalar/N=1024-16        	 3218389	       373.0 ns/op	32942.39 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32Scalar/N=8192-16        	  407226	      2924 ns/op	33618.78 MB/s	       0 B/op	       0 allocs/op
BenchmarkAxpyFloat32Scalar/N=65536-16       	   51741	     23181 ns/op	33925.16 MB/s	       0 B/op	       0 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	9.860s
```

### 2026-05-18 Metal softmax kernel expansion

This adds real Metal device kernels for last-dimension softmax:

- `softmax_{float32,float16,bfloat16}`

The kernels live in `pkg/backend/device/metal/softmax.metal` and run
through `pkg/backend/device/metal/bridge_softmax_darwin.m`. Each row
is assigned one 256-thread threadgroup. The kernel performs a parallel
max reduction, a parallel sum reduction over shifted exponentials, and
then normalized writes back to the same storage dtype. Float16 and
bfloat16 read their native storage, compute in float32, and write
their native storage. The parity cases use rows=13 and
`N ∈ {1, 7, 64, 1024, 8192}` as the softmax row width. F32 parity uses
a 64-ULP bound against the scalar reference because the device path
uses parallel reduction order plus Metal `exp`; f16 and bf16 parity
use a 1-ULP bound on the stored 16-bit representation.

The Metal dense registry now has 93 verified signatures: 57
elementwise, 27 shape, 6 matmul, and 3 softmax signatures.

Focused host softmax parity command:

```
go test ./pkg/backend/compute/kernels -run 'TestSoftmax(Float32|Float16AndBFloat16)$' -count=1 -v
=== RUN   TestSoftmaxFloat32
--- PASS: TestSoftmaxFloat32 (0.00s)
=== RUN   TestSoftmaxFloat16AndBFloat16
--- PASS: TestSoftmaxFloat16AndBFloat16 (0.00s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	0.532s
```

Focused Metal matmul + softmax parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_Metal(Softmax|MatMul)' -count=1 -v
=== RUN   TestKernelRegistry_MetalMatMulDTypes
--- PASS: TestKernelRegistry_MetalMatMulDTypes (0.06s)
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes
--- PASS: TestKernelRegistry_MetalMatMulAddDTypes (0.02s)
=== RUN   TestKernelRegistry_MetalSoftmaxDTypes
--- PASS: TestKernelRegistry_MetalSoftmaxDTypes (0.02s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.437s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	3.766s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	3.523s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	3.649s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	3.275s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	3.218s
```

Metal matmul + softmax benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_Run(Softmax|MatMul)DTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunMatMulDTypes/f32/matmul-16         	   10827	    108208 ns/op	1816.95 MB/s	    1346 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/f32/matmul_add-16     	   10000	    106145 ns/op	1857.08 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunMatMulDTypes/f16/matmul-16         	    9666	    105796 ns/op	 929.18 MB/s	    1344 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/f16/matmul_add-16     	    9876	    109676 ns/op	 898.65 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunMatMulDTypes/bf16/matmul-16        	    9957	    107538 ns/op	 914.13 MB/s	    1344 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/bf16/matmul_add-16    	   10000	    107582 ns/op	 916.14 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunSoftmaxDTypes/f32-16               	   11238	    106679 ns/op	9829.28 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunSoftmaxDTypes/f16-16               	   10000	    106526 ns/op	4921.67 MB/s	    1304 B/op	       5 allocs/op
BenchmarkKernel_RunSoftmaxDTypes/bf16-16              	   10000	    108761 ns/op	4820.56 MB/s	    1304 B/op	       5 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	10.060s
```

### 2026-05-18 Metal matmul kernel expansion

This adds real tiled Metal device kernels for dense matrix
multiplication and fused matrix multiplication plus bias:

- `matmul_{float32,float16,bfloat16}`
- `matmul_add_{float32,float16,bfloat16}`

The kernels live in `pkg/backend/device/metal/matmul.metal` and run
through `pkg/backend/device/metal/bridge_matmul_darwin.m`. Each
threadgroup computes a 16×16 output tile, staging left and right tiles
through `threadgroup` memory. `float32` reads and writes float storage;
`float16` and `bfloat16` read their native storage, accumulate in
float32, then write back to the same storage dtype. The parity cases
use rows=17 and cols=19 so every run exercises multiple threadgroups
and partial boundary tiles while N drives the inner dimension at
`N ∈ {1, 7, 64, 1024, 8192}`. F32 parity uses a 1-ULP bound; f16 and
bf16 parity use a 1-ULP bound on the stored 16-bit representation.

After this slice, the Metal dense registry had 90 verified signatures: 57
elementwise, 27 shape, and 6 matmul signatures.

Focused matmul parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_MetalMatMul' -count=1 -v
=== RUN   TestKernelRegistry_MetalMatMulDTypes
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f32
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f32/N=1
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f32/N=7
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f32/N=64
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f32/N=1024
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f32/N=8192
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f16
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f16/N=1
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f16/N=7
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f16/N=64
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f16/N=1024
=== RUN   TestKernelRegistry_MetalMatMulDTypes/f16/N=8192
=== RUN   TestKernelRegistry_MetalMatMulDTypes/bf16
=== RUN   TestKernelRegistry_MetalMatMulDTypes/bf16/N=1
=== RUN   TestKernelRegistry_MetalMatMulDTypes/bf16/N=7
=== RUN   TestKernelRegistry_MetalMatMulDTypes/bf16/N=64
=== RUN   TestKernelRegistry_MetalMatMulDTypes/bf16/N=1024
=== RUN   TestKernelRegistry_MetalMatMulDTypes/bf16/N=8192
--- PASS: TestKernelRegistry_MetalMatMulDTypes (0.05s)
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f32
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f32/N=1
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f32/N=7
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f32/N=64
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f32/N=1024
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f32/N=8192
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f16
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f16/N=1
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f16/N=7
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f16/N=64
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f16/N=1024
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/f16/N=8192
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/bf16
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/bf16/N=1
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/bf16/N=7
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/bf16/N=64
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/bf16/N=1024
=== RUN   TestKernelRegistry_MetalMatMulAddDTypes/bf16/N=8192
--- PASS: TestKernelRegistry_MetalMatMulAddDTypes (0.02s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.453s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	1.076s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	1.032s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.385s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	1.438s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	2.215s
```

Metal matmul benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunMatMulDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunMatMulDTypes/f32/matmul-16  	    9880	    130809 ns/op	1503.01 MB/s	    1345 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/f32/matmul_add-16         	    8084	    125851 ns/op	1566.30 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunMatMulDTypes/f16/matmul-16             	    9843	    115594 ns/op	 850.43 MB/s	    1344 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/f16/matmul_add-16         	    9355	    116103 ns/op	 848.90 MB/s	    1360 B/op	       8 allocs/op
BenchmarkKernel_RunMatMulDTypes/bf16/matmul-16            	    9951	    117786 ns/op	 834.60 MB/s	    1344 B/op	       7 allocs/op
BenchmarkKernel_RunMatMulDTypes/bf16/matmul_add-16        	   10000	    117916 ns/op	 835.85 MB/s	    1360 B/op	       8 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	7.260s
```

During the final verification pass, `pkg/backend/compute/kernels/elementwise_f32_neon_arm64.s`
blocked builds because the Go assembler does not accept its vector
float32 mnemonics. I verified the S4 encodings locally with
`clang -target arm64-apple-macos` plus `otool -X -s __TEXT __text` and
replaced those mnemonics with explicit NEON `WORD` macros. The focused
package sweep above covers the corrected assembly file.

### 2026-05-18 Metal shape kernel expansion

This adds real Metal device kernels for dense shape/data-movement ops
across `float32`, `float16`, and `bfloat16` storage:

- `reshape`, `merge_heads`, `split_heads`, `view_as_heads`
- `concat`, `split2`, `last_token`, `transpose2d`
- `upsample_nearest2d`

The kernels live in `pkg/backend/device/metal/shape.metal` and run
through the Objective-C shape bridge. The registry was already split
by dtype in `pkg/backend/device/metal/shape.go`; this update makes the
Metal shader entry points dtype-specific as well:

- `copy_{float32,float16,bfloat16}`
- `concat_{float32,float16,bfloat16}`
- `split2_{float32,float16,bfloat16}`
- `last_token_{float32,float16,bfloat16}`
- `transpose2d_{float32,float16,bfloat16}`
- `upsample_nearest2d_{float32,float16,bfloat16}`

`copy`, `concat`, `split2`, and `last_token` use `uint4` 16-byte
movement for contiguous aligned ranges and copy individual bytes only
for tail or boundary ranges. `transpose2d` and `upsample_nearest2d`
use exact-width storage kernels: `uint` for `float32`, `ushort` for
`float16`, and `ushort` for `bfloat16`, preserving tensor bytes exactly
instead of converting numeric values. `split2` uses the completion
registry's multi-output path so both destination tensors become ready
from the same Metal command completion. Shape dispatch resolves dtype
before registering the async completion token, so a validation error
cannot leave a pending tensor use-count.

Focused shape parity command:

```
go test ./pkg/backend/device/metal -run TestKernelRegistry_MetalShapeDTypes -count=1 -v
=== RUN   TestKernelRegistry_MetalShapeDTypes
--- PASS: TestKernelRegistry_MetalShapeDTypes (0.07s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.567s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.601s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.498s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.844s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	1.027s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.402s
```

Metal shape benchmark output:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunShapeDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunShapeDTypes/f32/reshape-16 	   10726	    110750 ns/op	 591.75 MB/s	    1321 B/op	       5 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/merge_heads-16         	   10000	    108388 ns/op	3627.84 MB/s	    1376 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/split_heads-16         	   10000	    108210 ns/op	3633.83 MB/s	    1376 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/view_as_heads-16       	   10000	    106448 ns/op	3693.98 MB/s	    1396 B/op	       8 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/concat-16              	   10000	    105819 ns/op	1238.64 MB/s	    1344 B/op	       5 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/split2-16              	   10000	    105713 ns/op	1239.89 MB/s	    1464 B/op	       6 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/last_token-16          	   10000	    105333 ns/op	2488.71 MB/s	    1368 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/transpose2d-16         	   10000	    108084 ns/op	1212.69 MB/s	    1368 B/op	       8 allocs/op
BenchmarkKernel_RunShapeDTypes/f32/upsample_nearest2d-16  	   10000	    111278 ns/op	2944.70 MB/s	    1448 B/op	       9 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/reshape-16             	   10000	    109502 ns/op	 299.25 MB/s	    1320 B/op	       5 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/merge_heads-16         	   10000	    109713 ns/op	1792.01 MB/s	    1376 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/split_heads-16         	   10000	    110099 ns/op	1785.74 MB/s	    1376 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/view_as_heads-16       	   10000	    109484 ns/op	1795.77 MB/s	    1396 B/op	       8 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/concat-16              	   10000	    109316 ns/op	 599.51 MB/s	    1344 B/op	       5 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/split2-16              	   10000	    110383 ns/op	 593.71 MB/s	    1464 B/op	       6 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/last_token-16          	    9870	    109620 ns/op	1195.70 MB/s	    1368 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/transpose2d-16         	   10000	    109599 ns/op	 597.96 MB/s	    1368 B/op	       8 allocs/op
BenchmarkKernel_RunShapeDTypes/f16/upsample_nearest2d-16  	   10000	    111559 ns/op	1468.64 MB/s	    1448 B/op	       9 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/reshape-16            	   10000	    110003 ns/op	 297.88 MB/s	    1320 B/op	       5 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/merge_heads-16        	   10000	    108453 ns/op	1812.84 MB/s	    1376 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/split_heads-16        	   10000	    109850 ns/op	1789.79 MB/s	    1376 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/view_as_heads-16      	   10000	    109479 ns/op	1795.85 MB/s	    1396 B/op	       8 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/concat-16             	    9958	    109049 ns/op	 600.98 MB/s	    1344 B/op	       5 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/split2-16             	   10000	    110424 ns/op	 593.49 MB/s	    1464 B/op	       6 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/last_token-16         	    9710	    110356 ns/op	1187.72 MB/s	    1368 B/op	       7 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/transpose2d-16        	   10000	    109619 ns/op	 597.85 MB/s	    1368 B/op	       8 allocs/op
BenchmarkKernel_RunShapeDTypes/bf16/upsample_nearest2d-16 	   10000	    111051 ns/op	1475.36 MB/s	    1448 B/op	       9 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	29.776s
```

### 2026-05-18 Metal elementwise dtype expansion

This adds real Metal device kernels for dense `float16` and `bfloat16`
elementwise execution across the same 19-operation surface already
verified for `float32`:

- Binary: `add`, `sub`, `mul`, `div`, `max`, `min`, `eq`, `ne`, `lt`,
  `le`, `gt`, `ge`.
- Unary: `relu`, `abs`, `neg`, `square`, `recip`, `sqrt`, `sign`.

`pkg/backend/device/metal/elementwise_float16.metal` uses native
`half4` vector kernels. `pkg/backend/device/metal/elementwise_bfloat16.metal`
stores BF16 values as `ushort4`, decodes on device by shifting into the
high float32 bits, performs the operation on device, and writes BF16
bits back through the same high-bit encoding used by `pkg/dtype`.
`pkg/backend/device/metal/bridge_elementwise_darwin.m` dispatches by
operation and dtype through the real Metal command queue. The kernel
registry now has 57 Metal dense elementwise signatures: 19 each for
`float32`, `float16`, and `bfloat16`.

Focused dtype parity command:

```
go test ./pkg/backend/device/metal -run 'TestKernelRegistry_Metal(Binary|Unary)ElementwiseDTypes' -count=1 -v
=== RUN   TestKernelRegistry_MetalBinaryElementwiseDTypes
--- PASS: TestKernelRegistry_MetalBinaryElementwiseDTypes (0.08s)
=== RUN   TestKernelRegistry_MetalUnaryElementwiseDTypes
--- PASS: TestKernelRegistry_MetalUnaryElementwiseDTypes (0.02s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.702s
```

Focused package sweep:

```
go test ./pkg/backend/device/metal/... ./pkg/backend/device/cuda ./pkg/backend/device/xla ./pkg/backend/compute/kernels -count=1
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.557s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	0.499s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	0.810s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	1.350s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.045s
```

Metal dtype benchmark output, `N=8192`:

```
go test ./pkg/backend/device/metal -run '^$' -bench 'BenchmarkKernel_RunElementwiseDTypes' -benchmem -count=1
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkKernel_RunElementwiseDTypes/f16/add-16     	   10882	    108345 ns/op	 453.66 MB/s	    1289 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/sub-16     	   10000	    103366 ns/op	 475.51 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/mul-16     	   10000	    101763 ns/op	 483.01 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/div-16     	   10000	    101096 ns/op	 486.19 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/max-16     	   10000	    101070 ns/op	 486.32 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/min-16     	   10000	    101773 ns/op	 482.96 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/eq-16      	   10000	    100825 ns/op	 487.50 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/ne-16      	   10000	    101475 ns/op	 484.38 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/lt-16      	   10000	    101086 ns/op	 486.24 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/le-16      	   10000	    101109 ns/op	 486.13 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/gt-16      	   10000	    101608 ns/op	 483.74 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/ge-16      	   10000	    101274 ns/op	 485.34 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/relu-16    	   10000	    101722 ns/op	 322.13 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/abs-16     	   10000	    101402 ns/op	 323.15 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/neg-16     	   10000	    101836 ns/op	 321.77 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/square-16  	   10000	    101702 ns/op	 322.20 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/recip-16   	   10000	    102519 ns/op	 319.63 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/sqrt-16    	   10000	    101228 ns/op	 323.70 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/f16/sign-16    	   10000	    102466 ns/op	 319.79 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/add-16    	   10000	    101648 ns/op	 483.55 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/sub-16    	   10000	    106007 ns/op	 463.67 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/mul-16    	   10000	    104784 ns/op	 469.08 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/div-16    	   10000	    101681 ns/op	 483.39 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/max-16    	   10000	    101253 ns/op	 485.44 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/min-16    	   10000	    101909 ns/op	 482.31 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/eq-16     	   10000	    102179 ns/op	 481.04 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/ne-16     	   10000	    101813 ns/op	 482.77 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/lt-16     	   10000	    101760 ns/op	 483.02 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/le-16     	   10000	    103094 ns/op	 476.77 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/gt-16     	   10000	    102155 ns/op	 481.15 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/ge-16     	   10000	    100543 ns/op	 488.87 MB/s	    1288 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/relu-16   	   10000	    107316 ns/op	 305.34 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/abs-16    	   10000	    101901 ns/op	 321.57 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/neg-16    	   10000	    102938 ns/op	 318.33 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/square-16 	   10000	    102073 ns/op	 321.02 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/recip-16  	   10000	    101383 ns/op	 323.21 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/sqrt-16   	   10000	    101065 ns/op	 324.23 MB/s	    1280 B/op	       3 allocs/op
BenchmarkKernel_RunElementwiseDTypes/bf16/sign-16   	   10000	    104843 ns/op	 312.54 MB/s	    1280 B/op	       3 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	39.495s
```

### 2026-05-18 Metal elementwise float32 expansion

This slice renames `pkg/backend/device/metal/add_float32.metal` to
`pkg/backend/device/metal/elementwise_float32.metal`, changes
`internal/metallibgen` to compile every `*.metal` source in the Metal
package, and expands the verified Metal device surface to 19 dense
float32 elementwise kernels:

- Binary: `add`, `sub`, `mul`, `div`, `max`, `min`, `eq`, `ne`, `lt`,
  `le`, `gt`, `ge`.
- Unary: `relu`, `abs`, `neg`, `square`, `recip`, `sqrt`, `sign`.

The focused parity command ran every operation at
`N ∈ {1, 7, 64, 1024, 8192}` through the real Metal command queue and
the device kernel registry. Exact arithmetic and comparison operations
assert bitwise float32 parity. `sqrt` asserts a 1-ULP bound against the
scalar reference.

Focused Metal device tests:

```
--- PASS: TestBackend_AddFloat32 (0.03s)
--- PASS: TestBackend_SubFloat32 (0.01s)
--- PASS: TestBackend_MulFloat32 (0.01s)
--- PASS: TestBackend_DivFloat32 (0.01s)
--- PASS: TestBackend_MaxFloat32 (0.01s)
--- PASS: TestBackend_MinFloat32 (0.01s)
--- PASS: TestBackend_EqFloat32 (0.01s)
--- PASS: TestBackend_NeFloat32 (0.01s)
--- PASS: TestBackend_LtFloat32 (0.01s)
--- PASS: TestBackend_LeFloat32 (0.01s)
--- PASS: TestBackend_GtFloat32 (0.01s)
--- PASS: TestBackend_GeFloat32 (0.01s)
--- PASS: TestKernelRegistry_MetalBinaryFloat32 (0.01s)
--- PASS: TestBackend_ReluFloat32 (0.01s)
--- PASS: TestBackend_AbsFloat32 (0.01s)
--- PASS: TestBackend_NegFloat32 (0.01s)
--- PASS: TestBackend_SquareFloat32 (0.01s)
--- PASS: TestBackend_RecipFloat32 (0.01s)
--- PASS: TestBackend_SqrtFloat32 (0.01s)
--- PASS: TestBackend_SignFloat32 (0.01s)
--- PASS: TestKernelRegistry_MetalUnaryFloat32 (0.01s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.787s
```

Focused package sweep:

```
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.550s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	1.406s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	1.696s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	0.663s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	1.070s
```

Metal benchmark output, `N=8192` rows from the full run:

```
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkNewBackend-16                          8132    150014 ns/op    1264 B/op   4 allocs/op
BenchmarkBackend_BinaryFloat32/add/N=8192-16   10000    105359 ns/op   933.04 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/sub/N=8192-16   10000    105312 ns/op   933.45 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/mul/N=8192-16   10000    106746 ns/op   920.91 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/div/N=8192-16    9921    106045 ns/op   927.00 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/max/N=8192-16   10000    106148 ns/op   926.10 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/min/N=8192-16   10000    110796 ns/op   887.25 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/eq/N=8192-16    10000    112528 ns/op   873.60 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/ne/N=8192-16     9360    119004 ns/op   826.06 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/lt/N=8192-16    10000    118599 ns/op   828.88 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/le/N=8192-16     9578    119903 ns/op   819.87 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/gt/N=8192-16     9302    126539 ns/op   776.87 MB/s  1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/ge/N=8192-16     9154    123562 ns/op   795.58 MB/s  1544 B/op   7 allocs/op
BenchmarkKernel_RunBinaryFloat32/add/N=8192-16 10000   120620 ns/op   814.99 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/sub/N=8192-16 10000   111386 ns/op   882.56 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/mul/N=8192-16 10000   118049 ns/op   832.74 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/div/N=8192-16 10000   109402 ns/op   898.56 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/max/N=8192-16 10000   107769 ns/op   912.17 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/min/N=8192-16 10000   110759 ns/op   887.55 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/eq/N=8192-16   9488   112988 ns/op   870.04 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/ne/N=8192-16  10000   116214 ns/op   845.89 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/lt/N=8192-16  10000   118660 ns/op   828.45 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/le/N=8192-16  10000   114773 ns/op   856.51 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/gt/N=8192-16  10000   114733 ns/op   856.80 MB/s  1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/ge/N=8192-16   9666   118899 ns/op   826.79 MB/s  1288 B/op   3 allocs/op
BenchmarkBackend_UnaryFloat32/relu/N=8192-16    9728   111825 ns/op   586.06 MB/s  1520 B/op   5 allocs/op
BenchmarkBackend_UnaryFloat32/abs/N=8192-16    10000   114219 ns/op   573.78 MB/s  1520 B/op   5 allocs/op
BenchmarkBackend_UnaryFloat32/neg/N=8192-16    10000   111727 ns/op   586.57 MB/s  1520 B/op   5 allocs/op
BenchmarkBackend_UnaryFloat32/square/N=8192-16 10000   111419 ns/op   588.19 MB/s  1520 B/op   5 allocs/op
BenchmarkBackend_UnaryFloat32/recip/N=8192-16  10000   108610 ns/op   603.41 MB/s  1520 B/op   5 allocs/op
BenchmarkBackend_UnaryFloat32/sqrt/N=8192-16   10000   108271 ns/op   605.30 MB/s  1520 B/op   5 allocs/op
BenchmarkBackend_UnaryFloat32/sign/N=8192-16   10000   114143 ns/op   574.16 MB/s  1520 B/op   5 allocs/op
BenchmarkKernel_RunUnaryFloat32/relu/N=8192-16 10000   117148 ns/op   559.43 MB/s  1280 B/op   3 allocs/op
BenchmarkKernel_RunUnaryFloat32/abs/N=8192-16  10000   114711 ns/op   571.31 MB/s  1280 B/op   3 allocs/op
BenchmarkKernel_RunUnaryFloat32/neg/N=8192-16  10000   116186 ns/op   564.06 MB/s  1280 B/op   3 allocs/op
BenchmarkKernel_RunUnaryFloat32/square/N=8192-16 10000 118529 ns/op  552.91 MB/s  1280 B/op   3 allocs/op
BenchmarkKernel_RunUnaryFloat32/recip/N=8192-16 10000  116137 ns/op  564.30 MB/s  1280 B/op   3 allocs/op
BenchmarkKernel_RunUnaryFloat32/sqrt/N=8192-16 10000   114455 ns/op  572.59 MB/s  1280 B/op   3 allocs/op
BenchmarkKernel_RunUnaryFloat32/sign/N=8192-16 10000   110328 ns/op  594.01 MB/s  1280 B/op   3 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	216.585s
```

### 2026-05-18 Metal binary float32 kernel slice

This slice verifies the Metal device kernels for dense float32
`add`, `sub`, `mul`, and `div`. Each operation runs through the real
Metal submission path, uses a `float4` vectorized body with scalar tail
handling, is registered in the device kernel registry, and is tested
against the scalar arithmetic reference with bitwise float32 parity at
`N ∈ {1, 7, 64, 1024, 8192}`.

The Objective-C pipeline cache now avoids holding its global `NSLock`
while `newComputePipelineStateWithFunction` compiles a pipeline. The
cache performs a locked read, compiles outside the cache lock, then
rechecks and inserts under the lock. This keeps independent Metal
kernel/dtype pipeline compilation from serializing on one dictionary
lock as the kernel surface expands.

`go generate ./pkg/backend/device/metal` completed and rebuilt
`pkg/backend/device/metal/kernels.metallib`.

Focused Metal device tests:

```
=== RUN   TestBackend_AddFloat32
=== RUN   TestBackend_AddFloat32/N=1

  Given two Metal float32 tensors for add ✔✔✔✔✔✔✔✔


8 total assertions

=== RUN   TestBackend_AddFloat32/N=7

  Given two Metal float32 tensors for add ✔✔✔✔✔✔✔✔


16 total assertions

=== RUN   TestBackend_AddFloat32/N=64

  Given two Metal float32 tensors for add ✔✔✔✔✔✔✔✔


24 total assertions

=== RUN   TestBackend_AddFloat32/N=1024

  Given two Metal float32 tensors for add ✔✔✔✔✔✔✔✔


32 total assertions

=== RUN   TestBackend_AddFloat32/N=8192

  Given two Metal float32 tensors for add ✔✔✔✔✔✔✔✔


40 total assertions

--- PASS: TestBackend_AddFloat32 (0.05s)
    --- PASS: TestBackend_AddFloat32/N=1 (0.01s)
    --- PASS: TestBackend_AddFloat32/N=7 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=64 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=1024 (0.00s)
    --- PASS: TestBackend_AddFloat32/N=8192 (0.00s)
=== RUN   TestBackend_SubFloat32
=== RUN   TestBackend_SubFloat32/N=1

  Given two Metal float32 tensors for sub ✔✔✔✔✔✔✔✔


48 total assertions

=== RUN   TestBackend_SubFloat32/N=7

  Given two Metal float32 tensors for sub ✔✔✔✔✔✔✔✔


56 total assertions

=== RUN   TestBackend_SubFloat32/N=64

  Given two Metal float32 tensors for sub ✔✔✔✔✔✔✔✔


64 total assertions

=== RUN   TestBackend_SubFloat32/N=1024

  Given two Metal float32 tensors for sub ✔✔✔✔✔✔✔✔


72 total assertions

=== RUN   TestBackend_SubFloat32/N=8192

  Given two Metal float32 tensors for sub ✔✔✔✔✔✔✔✔


80 total assertions

--- PASS: TestBackend_SubFloat32 (0.01s)
    --- PASS: TestBackend_SubFloat32/N=1 (0.00s)
    --- PASS: TestBackend_SubFloat32/N=7 (0.00s)
    --- PASS: TestBackend_SubFloat32/N=64 (0.00s)
    --- PASS: TestBackend_SubFloat32/N=1024 (0.00s)
    --- PASS: TestBackend_SubFloat32/N=8192 (0.00s)
=== RUN   TestBackend_MulFloat32
=== RUN   TestBackend_MulFloat32/N=1

  Given two Metal float32 tensors for mul ✔✔✔✔✔✔✔✔


88 total assertions

=== RUN   TestBackend_MulFloat32/N=7

  Given two Metal float32 tensors for mul ✔✔✔✔✔✔✔✔


96 total assertions

=== RUN   TestBackend_MulFloat32/N=64

  Given two Metal float32 tensors for mul ✔✔✔✔✔✔✔✔


104 total assertions

=== RUN   TestBackend_MulFloat32/N=1024

  Given two Metal float32 tensors for mul ✔✔✔✔✔✔✔✔


112 total assertions

=== RUN   TestBackend_MulFloat32/N=8192

  Given two Metal float32 tensors for mul ✔✔✔✔✔✔✔✔


120 total assertions

--- PASS: TestBackend_MulFloat32 (0.01s)
    --- PASS: TestBackend_MulFloat32/N=1 (0.00s)
    --- PASS: TestBackend_MulFloat32/N=7 (0.00s)
    --- PASS: TestBackend_MulFloat32/N=64 (0.00s)
    --- PASS: TestBackend_MulFloat32/N=1024 (0.00s)
    --- PASS: TestBackend_MulFloat32/N=8192 (0.00s)
=== RUN   TestBackend_DivFloat32
=== RUN   TestBackend_DivFloat32/N=1

  Given two Metal float32 tensors for div ✔✔✔✔✔✔✔✔


128 total assertions

=== RUN   TestBackend_DivFloat32/N=7

  Given two Metal float32 tensors for div ✔✔✔✔✔✔✔✔


136 total assertions

=== RUN   TestBackend_DivFloat32/N=64

  Given two Metal float32 tensors for div ✔✔✔✔✔✔✔✔


144 total assertions

=== RUN   TestBackend_DivFloat32/N=1024

  Given two Metal float32 tensors for div ✔✔✔✔✔✔✔✔


152 total assertions

=== RUN   TestBackend_DivFloat32/N=8192

  Given two Metal float32 tensors for div ✔✔✔✔✔✔✔✔


160 total assertions

--- PASS: TestBackend_DivFloat32 (0.01s)
    --- PASS: TestBackend_DivFloat32/N=1 (0.00s)
    --- PASS: TestBackend_DivFloat32/N=7 (0.00s)
    --- PASS: TestBackend_DivFloat32/N=64 (0.00s)
    --- PASS: TestBackend_DivFloat32/N=1024 (0.00s)
    --- PASS: TestBackend_DivFloat32/N=8192 (0.00s)
=== RUN   TestBackend_AddFloat32_CloseInputsBeforeDownload

  Given a queued Metal add whose inputs are closed immediately ✔✔✔✔✔✔✔


167 total assertions

--- PASS: TestBackend_AddFloat32_CloseInputsBeforeDownload (0.01s)
=== RUN   TestBackend_AddFloat32_CloseOutputBeforeCompletion

  Given a queued Metal add whose output is closed immediately ✔✔✔✔✔✔✔✔


175 total assertions

--- PASS: TestBackend_AddFloat32_CloseOutputBeforeCompletion (0.00s)
=== RUN   TestMetalBufferPool_AlignedBuckets

  Given closed Metal tensors with nearby byte sizes ✔✔✔✔✔✔✔✔✔


184 total assertions

--- PASS: TestMetalBufferPool_AlignedBuckets (0.00s)
=== RUN   TestKernelRegistry_MetalBinaryFloat32
=== RUN   TestKernelRegistry_MetalBinaryFloat32/add

  Given the device kernel registry for add ✔✔✔✔✔✔✔✔✔


193 total assertions

=== RUN   TestKernelRegistry_MetalBinaryFloat32/sub

  Given the device kernel registry for sub ✔✔✔✔✔✔✔✔✔


202 total assertions

=== RUN   TestKernelRegistry_MetalBinaryFloat32/mul

  Given the device kernel registry for mul ✔✔✔✔✔✔✔✔✔


211 total assertions

=== RUN   TestKernelRegistry_MetalBinaryFloat32/div

  Given the device kernel registry for div ✔✔✔✔✔✔✔✔✔


220 total assertions

--- PASS: TestKernelRegistry_MetalBinaryFloat32 (0.01s)
    --- PASS: TestKernelRegistry_MetalBinaryFloat32/add (0.00s)
    --- PASS: TestKernelRegistry_MetalBinaryFloat32/sub (0.00s)
    --- PASS: TestKernelRegistry_MetalBinaryFloat32/mul (0.00s)
    --- PASS: TestKernelRegistry_MetalBinaryFloat32/div (0.00s)
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.683s
```

Focused package sweep:

```
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	0.303s
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal/internal/metallibgen	1.180s
ok  	github.com/theapemachine/caramba/pkg/backend/device/cuda	1.015s
ok  	github.com/theapemachine/caramba/pkg/backend/device/xla	0.546s
ok  	github.com/theapemachine/caramba/pkg/backend/compute/kernels	0.782s
```

Metal benchmark output:

```
goos: darwin
goarch: arm64
pkg: github.com/theapemachine/caramba/pkg/backend/device/metal
cpu: Apple M4 Max
BenchmarkNewBackend-16                         7341    155629 ns/op     1264 B/op   4 allocs/op
BenchmarkBackend_BinaryFloat32/add/N=1-16     10140    117895 ns/op     0.10 MB/s   1545 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/add/N=7-16     10000    118611 ns/op     0.71 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/add/N=64-16     9830    118297 ns/op     6.49 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/add/N=1024-16  10000    116692 ns/op   105.30 MB/s   1545 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/add/N=8192-16   9982    120691 ns/op   814.51 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/sub/N=1-16      9492    121018 ns/op     0.10 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/sub/N=7-16      9714    122915 ns/op     0.68 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/sub/N=64-16     9253    121380 ns/op     6.33 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/sub/N=1024-16   9633    120253 ns/op   102.18 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/sub/N=8192-16   9810    122527 ns/op   802.30 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/mul/N=1-16      8665    121331 ns/op     0.10 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/mul/N=7-16      9733    120100 ns/op     0.70 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/mul/N=64-16     9867    119287 ns/op     6.44 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/mul/N=1024-16   9550    120303 ns/op   102.14 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/mul/N=8192-16   9765    120878 ns/op   813.25 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/div/N=1-16     10000    122046 ns/op     0.10 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/div/N=7-16      9562    126236 ns/op     0.67 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/div/N=64-16     9982    126900 ns/op     6.05 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/div/N=1024-16  10000    128496 ns/op    95.63 MB/s   1544 B/op   7 allocs/op
BenchmarkBackend_BinaryFloat32/div/N=8192-16   9286    128728 ns/op   763.65 MB/s   1544 B/op   7 allocs/op
BenchmarkKernel_RunBinaryFloat32/add/N=1-16    9736    123384 ns/op     0.10 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/add/N=7-16    9392    121116 ns/op     0.69 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/add/N=64-16  10000    119415 ns/op     6.43 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/add/N=1024-16 9709    120796 ns/op   101.73 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/add/N=8192-16 9867    117727 ns/op   835.02 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/sub/N=1-16   10000    114201 ns/op     0.11 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/sub/N=7-16   10000    120828 ns/op     0.70 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/sub/N=64-16  10000    120860 ns/op     6.35 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/sub/N=1024-16 10000   122830 ns/op   100.04 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/sub/N=8192-16 9744    122989 ns/op   799.29 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/mul/N=1-16    9408    122922 ns/op     0.10 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/mul/N=7-16    9908    122586 ns/op     0.69 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/mul/N=64-16   9520    121190 ns/op     6.34 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/mul/N=1024-16 9978    120994 ns/op   101.56 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/mul/N=8192-16 9164    121081 ns/op   811.89 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/div/N=1-16   10000    118541 ns/op     0.10 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/div/N=7-16   10000    121127 ns/op     0.69 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/div/N=64-16  10000    120437 ns/op     6.38 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/div/N=1024-16 10000   121434 ns/op   101.19 MB/s   1288 B/op   3 allocs/op
BenchmarkKernel_RunBinaryFloat32/div/N=8192-16 9954    118273 ns/op   831.16 MB/s   1288 B/op   3 allocs/op
PASS
ok  	github.com/theapemachine/caramba/pkg/backend/device/metal	48.748s
```

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
| 4     | Metal device backend          | 195 verified dense elementwise + shape + matmul + softmax + normalization + projection/model + transformer embedding/masking + vision signatures for `float32`, `float16`, `bfloat16` |
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

## Phase 4 / 5 / 6: device backends

| file                                          | status                |
| --------------------------------------------- | --------------------- |
| `pkg/backend/device/metal/backend.go`         | verified              |
| `pkg/backend/device/metal/bridge_stub.go`     | verified              |
| `pkg/backend/device/metal/bridge_darwin.go`   | verified              |
| `pkg/backend/device/metal/bridge_darwin.h`    | verified              |
| `pkg/backend/device/metal/bridge_darwin.m`    | verified              |
| `pkg/backend/device/metal/bridge_darwin_private.h` | verified        |
| `pkg/backend/device/metal/bridge_elementwise_darwin.m` | verified     |
| `pkg/backend/device/metal/bridge_matmul_darwin.m` | verified        |
| `pkg/backend/device/metal/bridge_normalization_darwin.m` | verified |
| `pkg/backend/device/metal/bridge_projection_darwin.m` | verified |
| `pkg/backend/device/metal/bridge_shape_common_darwin.m` | verified    |
| `pkg/backend/device/metal/bridge_shape_darwin.m` | verified           |
| `pkg/backend/device/metal/bridge_shape_private.h` | verified          |
| `pkg/backend/device/metal/bridge_softmax_darwin.m` | verified       |
| `pkg/backend/device/metal/bridge_transformer_darwin.m` | verified |
| `pkg/backend/device/metal/bridge_transformer_masking_darwin.m` | verified |
| `pkg/backend/device/metal/bridge_transformer_private.h` | verified |
| `pkg/backend/device/metal/bridge_unary_darwin.m` | verified           |
| `pkg/backend/device/metal/bridge_vision_convolution_darwin.m` | verified |
| `pkg/backend/device/metal/bridge_vision_darwin.m` | verified       |
| `pkg/backend/device/metal/bridge_vision_private.h` | verified      |
| `pkg/backend/device/metal/unary_darwin.go`    | verified              |
| `pkg/backend/device/metal/elementwise_float32.metal` | verified       |
| `pkg/backend/device/metal/elementwise_float16.metal` | verified       |
| `pkg/backend/device/metal/elementwise_bfloat16.metal` | verified     |
| `pkg/backend/device/metal/elementwise_extended.metal` | verified      |
| `pkg/backend/device/metal/matmul.metal`       | verified              |
| `pkg/backend/device/metal/normalization.metal` | verified             |
| `pkg/backend/device/metal/projection.metal`   | verified              |
| `pkg/backend/device/metal/shape.metal`        | verified              |
| `pkg/backend/device/metal/softmax.metal`      | verified              |
| `pkg/backend/device/metal/transformer.metal`  | verified              |
| `pkg/backend/device/metal/vision.metal`       | verified              |
| `pkg/backend/device/metal/elementwise_dtype_darwin.go` | verified    |
| `pkg/backend/device/metal/elementwise_dtype_test.go` | verified       |
| `pkg/backend/device/metal/elementwise_dtype_bench_test.go` | verified |
| `pkg/backend/device/metal/elementwise_unary_extended_test.go` | verified |
| `pkg/backend/device/metal/elementwise_unary_extended_bench_test.go` | verified |
| `pkg/backend/device/metal/kernels.metallib`   | verified              |
| `pkg/backend/device/metal/generate.go`        | verified              |
| `pkg/backend/device/metal/kernels.go`         | verified              |
| `pkg/backend/device/metal/matmul.go`          | verified              |
| `pkg/backend/device/metal/matmul_darwin.go`   | verified              |
| `pkg/backend/device/metal/matmul_stub.go`     | verified              |
| `pkg/backend/device/metal/matmul_test.go`     | verified              |
| `pkg/backend/device/metal/matmul_bench_test.go` | verified            |
| `pkg/backend/device/metal/normalization.go`   | verified              |
| `pkg/backend/device/metal/normalization_darwin.go` | verified          |
| `pkg/backend/device/metal/normalization_stub.go` | verified            |
| `pkg/backend/device/metal/normalization_test.go` | verified            |
| `pkg/backend/device/metal/normalization_test_helpers.go` | verified    |
| `pkg/backend/device/metal/normalization_bench_test.go` | verified      |
| `pkg/backend/device/metal/projection.go`      | verified              |
| `pkg/backend/device/metal/projection_darwin.go` | verified            |
| `pkg/backend/device/metal/projection_test.go` | verified              |
| `pkg/backend/device/metal/projection_bench_test.go` | verified        |
| `pkg/backend/device/metal/transformer.go`     | verified              |
| `pkg/backend/device/metal/transformer_darwin.go` | verified           |
| `pkg/backend/device/metal/transformer_validate_darwin.go` | verified |
| `pkg/backend/device/metal/transformer_embedding_test.go` | verified |
| `pkg/backend/device/metal/transformer_masking_test.go` | verified |
| `pkg/backend/device/metal/transformer_bench_test.go` | verified     |
| `pkg/backend/device/metal/vision.go`          | verified              |
| `pkg/backend/device/metal/vision_convolution_darwin.go` | verified |
| `pkg/backend/device/metal/vision_convolution_test.go` | verified |
| `pkg/backend/device/metal/vision_convolution_expected_test.go` | verified |
| `pkg/backend/device/metal/vision_darwin.go`   | verified              |
| `pkg/backend/device/metal/vision_test.go`     | verified              |
| `pkg/backend/device/metal/vision_expected_test.go` | verified       |
| `pkg/backend/device/metal/vision_bench_test.go` | verified            |
| `pkg/backend/device/metal/shape.go`           | verified              |
| `pkg/backend/device/metal/shape_darwin.go`    | verified              |
| `pkg/backend/device/metal/shape_validate_darwin.go` | verified       |
| `pkg/backend/device/metal/shape_test.go`      | verified              |
| `pkg/backend/device/metal/shape_test_helpers.go` | verified          |
| `pkg/backend/device/metal/shape_bench_test.go` | verified             |
| `pkg/backend/device/metal/softmax.go`         | verified              |
| `pkg/backend/device/metal/softmax_darwin.go`  | verified              |
| `pkg/backend/device/metal/softmax_stub.go`    | verified              |
| `pkg/backend/device/metal/softmax_test.go`    | verified              |
| `pkg/backend/device/metal/softmax_bench_test.go` | verified           |
| `pkg/backend/device/metal/unary_float32.go`   | verified              |
| `pkg/backend/device/metal/unary_float32_test.go` | verified           |
| `pkg/backend/device/metal/unary_extended.go`  | verified              |
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
| `pkg/backend/compute/kernels/softmax_dtype_test.go` | verified |
| `pkg/backend/compute/kernels/layernorm.go`        | verified  |
| `pkg/backend/compute/kernels/layernorm_dtype.go`  | verified  |
| `pkg/backend/compute/kernels/layernorm_dtype_test.go` | verified |
| `pkg/backend/compute/kernels/axpy_f32_dispatch_arm64.go` | verified |
| `pkg/backend/compute/kernels/axpy_f32_dispatch_other.go` | verified |
| `pkg/backend/compute/kernels/axpy_f32_neon_arm64.s` | verified |
| `pkg/backend/compute/kernels/axpy_f32_neon_arm64.go` | verified |
| `pkg/backend/compute/kernels/axpy_f32_neon_arm64_test.go` | verified |
| `pkg/backend/compute/kernels/kernels_test.go`     | verified  |
| `pkg/backend/compute/kernels/matmul_test.go`      | verified  |

Still pending: attention variants (flash-attention, rope,
sliding window), optimizer states (Adam/AdamW/Lion/Sophia), quantized
inference kernels (GPTQ, AWQ, SmoothQuant), FP8 paths, and SIMD `.s`
bodies for every kernel shipped today.

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
