package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Attention variants beyond the basic dense kernel and flash-attention:
multi-head attention with per-head splits, grouped-query attention
(GQA), multi-query attention (MQA), sliding-window attention, and
ALiBi-biased attention.

The kernel signatures fold the variant-specific scalars into config
structs so the dispatcher signature remains (Q, K, V, output).

Per Phase 8.2, batched attention requires extending the dispatch
table to (Q, K, V, output, mask) — that lands in a follow-up;
the kernels here apply masking via the config rather than an input
tensor for now.
*/

type MultiHeadAttentionConfig struct {
	NumHeads    int
	HeadDim     int
	Causal      bool
	WindowSize  int     // 0 = no window
	ALiBiSlope  float32 // 0 = disabled
	KVHeadCount int     // for GQA/MQA; 0 → equals NumHeads (full multi-head)
}

func DefaultMultiHeadAttentionConfig() MultiHeadAttentionConfig {
	return MultiHeadAttentionConfig{NumHeads: 8, HeadDim: 64}
}

func init() {
	Default.Register(Kernel{
		Name: "multi_head_attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMultiHeadAttentionDefault,
	})

	Default.Register(Kernel{
		Name: "grouped_query_attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runGroupedQueryAttentionDefault,
	})

	Default.Register(Kernel{
		Name: "sliding_window_attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSlidingWindowAttentionDefault,
	})
}

func runMultiHeadAttentionDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return MultiHeadAttentionFloat32(
		DefaultMultiHeadAttentionConfig(),
		args[0], args[1], args[2], args[3],
	)
}

func runGroupedQueryAttentionDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	config := DefaultMultiHeadAttentionConfig()
	config.KVHeadCount = config.NumHeads / 4 // typical GQA ratio

	return MultiHeadAttentionFloat32(config, args[0], args[1], args[2], args[3])
}

func runSlidingWindowAttentionDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	config := DefaultMultiHeadAttentionConfig()
	config.Causal = true
	config.WindowSize = 128

	return MultiHeadAttentionFloat32(config, args[0], args[1], args[2], args[3])
}

/*
MultiHeadAttentionFloat32 splits Q/K/V along the head dimension and
runs the attention computation per head. Supports causal masking,
sliding-window masking, ALiBi bias, and GQA/MQA via KVHeadCount < NumHeads
(each KV head is shared across NumHeads / KVHeadCount query heads).

Shapes:
  - query  [seqQ, numHeads * headDim]
  - key    [seqK, kvHeadCount * headDim]
  - value  [seqK, kvHeadCount * headDim]
  - output [seqQ, numHeads * headDim]
*/
func MultiHeadAttentionFloat32(
	config MultiHeadAttentionConfig,
	query, key, value, out tensor.Tensor,
) error {
	queryDims := query.Shape().Dims()
	keyDims := key.Shape().Dims()
	valueDims := value.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 ||
		len(valueDims) != 2 || len(outDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	queryFeatures := config.NumHeads * config.HeadDim
	kvFeatures := kvHeads * config.HeadDim

	if queryDims[1] != queryFeatures || keyDims[1] != kvFeatures ||
		valueDims[1] != kvFeatures || outDims[1] != queryFeatures {
		return tensor.ErrShapeMismatch
	}

	seqQ := queryDims[0]
	seqK := keyDims[0]

	if valueDims[0] != seqK || outDims[0] != seqQ {
		return tensor.ErrShapeMismatch
	}

	queryView, _ := query.Float32Native()
	keyView, _ := key.Float32Native()
	valueView, _ := value.Float32Native()
	outView, _ := out.Float32Native()

	multiHeadAttentionSlices(config, queryView, keyView, valueView, outView, seqQ, seqK, kvHeads)
	return nil
}

func multiHeadAttentionSlices(
	config MultiHeadAttentionConfig,
	queryView, keyView, valueView, outView []float32,
	seqQ, seqK, kvHeads int,
) {
	scale := float32(1.0 / math.Sqrt(float64(config.HeadDim)))
	headsPerKVHead := config.NumHeads / kvHeads

	for headIndex := 0; headIndex < config.NumHeads; headIndex++ {
		kvHeadIndex := headIndex / headsPerKVHead

		runSingleHead(
			queryView, keyView, valueView, outView,
			seqQ, seqK,
			config.HeadDim, config.NumHeads, kvHeads,
			headIndex, kvHeadIndex,
			scale, config,
		)
	}
}

func runSingleHead(
	queryView, keyView, valueView, outView []float32,
	seqQ, seqK, headDim, numHeads, kvHeads, headIndex, kvHeadIndex int,
	scale float32,
	config MultiHeadAttentionConfig,
) {
	queryHeadOffset := headIndex * headDim
	kvHeadOffset := kvHeadIndex * headDim
	queryStride := numHeads * headDim
	kvStride := kvHeads * headDim

	scores := make([]float32, seqK)

	for qIndex := range seqQ {
		computeHeadScoresNative(
			queryView, keyView,
			qIndex, seqK, headDim,
			queryHeadOffset, kvHeadOffset,
			queryStride, kvStride,
			scale, scores,
			config,
		)
		stableSoftmaxRowNative(scores)
		writeHeadOutputNative(
			scores, valueView, outView,
			qIndex, seqK, headDim,
			queryHeadOffset, kvHeadOffset,
			queryStride, kvStride,
		)
	}
}
