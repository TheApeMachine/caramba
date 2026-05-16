//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/dispatch"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ executor.Backend = (*TensorBackend)(nil)

func (tensorBackend *TensorBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	switch executor.NormalizeOperation(node.Op) {
	case ir.OpInput:
		return nil, fmt.Errorf("metal tensor: input node %q was not materialized", node.ID)
	case ir.OpAdd:
		return requireMetalInputs(node, inputs, 2, tensorBackend.Add)
	case ir.OpMul:
		return requireMetalInputs(node, inputs, 2, tensorBackend.Mul)
	case ir.OpMatmul:
		return requireMetalInputs(node, inputs, 2, tensorBackend.Matmul)
	case ir.OpReLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.ReLU(input)
		})
	case ir.OpLeakyReLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.LeakyReLU(input, 0.01)
		})
	case ir.OpGELU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.GELU(input)
		})
	case ir.OpTanh:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Tanh(input)
		})
	case ir.OpSigmoid:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Sigmoid(input)
		})
	case ir.OpSwish:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Swish(input)
		})
	case ir.OpSELU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.SELU(input)
		})
	case ir.OpSwiGLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.SwiGLU(input)
		})
	case ir.OpFused:
		if len(inputs) != 3 {
			return nil, fmt.Errorf("metal tensor: Fused node %q requires 3 inputs", node.ID)
		}

		activation, _ := node.Metadata["activation"].(string)
		if strings.EqualFold(activation, string(ir.OpGELU)) {
			return tensorBackend.MatmulAddGELU(inputs[0], inputs[1], inputs[2])
		}

		return tensorBackend.MatmulAdd(inputs[0], inputs[1], inputs[2])
	default:
		return tensorBackend.applyModelOperation(ctx, node, inputs)
	}
}

func (tensorBackend *TensorBackend) ReLU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.ReLUTensor(input)
}

func (tensorBackend *TensorBackend) LeakyReLU(
	input tensor.Float64Tensor, alpha float64,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.LeakyReLUTensor(input, alpha)
}

func (tensorBackend *TensorBackend) GELU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.GELUTensor(input)
}

func (tensorBackend *TensorBackend) Tanh(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.TanhTensor(input)
}

func (tensorBackend *TensorBackend) Sigmoid(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.SigmoidTensor(input)
}

func (tensorBackend *TensorBackend) Swish(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.SwishTensor(input)
}

func (tensorBackend *TensorBackend) SELU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.SELUTensor(input)
}

func (tensorBackend *TensorBackend) SwiGLU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	activationOps, err := tensorBackend.activation()

	if err != nil {
		return nil, err
	}

	return activationOps.SwiGLUTensor(input)
}

func (tensorBackend *TensorBackend) Add(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.AddTensor(left, right)
}

func (tensorBackend *TensorBackend) Mul(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.MulTensor(left, right)
}

func (tensorBackend *TensorBackend) Matmul(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.MatmulTensor(left, right)
}

func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.MatmulAddTensor(left, right, bias)
}

func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.MatmulAddGELUTensor(left, right, bias)
}

func requireMetalInputs(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
	count int,
	apply func(tensor.Float64Tensor, tensor.Float64Tensor) (tensor.Float64Tensor, error),
) (tensor.Float64Tensor, error) {
	if len(inputs) != count {
		return nil, fmt.Errorf("metal tensor: %s node %q requires %d inputs", node.Op, node.ID, count)
	}

	var second tensor.Float64Tensor
	if len(inputs) > 1 {
		second = inputs[1]
	}

	return apply(inputs[0], second)
}

func (tensorBackend *TensorBackend) applyModelOperation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	switch strings.ToLower(string(node.Op)) {
	case "embedding.token":
		return tensorBackend.applyTokenEmbedding(ctx, node, inputs)
	case "math.rmsnorm":
		return tensorBackend.applyRMSNorm(ctx, node, inputs)
	case "math.layernorm":
		return tensorBackend.applyLayerNorm(ctx, node, inputs)
	case "shape.view_as_heads":
		return tensorBackend.applyViewAsHeads(ctx, node, inputs)
	case "shape.merge_heads":
		return tensorBackend.applyMergeHeads(ctx, node, inputs)
	case "shape.last_token":
		return tensorBackend.applyLastToken(ctx, node, inputs)
	case "projection.linear":
		return tensorBackend.applyLinear(ctx, node, inputs)
	case "attention.sdpa":
		return tensorBackend.applySDPA(ctx, node, inputs)
	case "attention.gqa":
		return tensorBackend.applyGQA(ctx, node, inputs)
	case "positional.rope":
		return tensorBackend.applyRoPE(ctx, node, inputs)
	default:
		return dispatch.RunOperation(
			ctx,
			tensorBackend,
			node,
			inputs,
			NewOperationRegistry(),
			NewOptimizerRegistry(),
		)
	}
}

func (tensorBackend *TensorBackend) applyTokenEmbedding(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: embedding node %q requires 1 input", node.ID)
	}

	vocabSize := intConfig(node, "vocab_size", 0)
	dModel := intConfig(node, "d_model", 0)

	if vocabSize <= 0 || dModel <= 0 {
		return nil, fmt.Errorf("metal tensor: embedding node %q requires vocab_size and d_model", node.ID)
	}

	weight, err := floatSliceConfigRequired(node, "weight")

	if err != nil {
		return nil, err
	}

	weightTensor, err := tensorBackend.cachedTensor(
		node.ID+":weight",
		[]int{vocabSize, dModel},
		weight,
	)

	if err != nil {
		return nil, err
	}

	embeddingOps, err := tensorBackend.embedding(vocabSize, dModel)

	if err != nil {
		return nil, err
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return embeddingOps.ForwardTensor(inputs[0], weightTensor, outputShape)
}

func (tensorBackend *TensorBackend) applyRMSNorm(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: rmsnorm node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("metal tensor: rmsnorm node %q input shape is required", node.ID)
	}

	dModel := inputShape[len(inputShape)-1]
	weight, err := floatSliceConfigRequired(node, "weight")

	if err != nil {
		return nil, err
	}

	weightTensor, err := tensorBackend.cachedTensor(node.ID+":weight", []int{dModel}, weight)

	if err != nil {
		return nil, err
	}

	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.RMSNormTensor(inputs[0], weightTensor, floatConfig(node, "eps", 1e-5))
}

func (tensorBackend *TensorBackend) applyLayerNorm(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: layernorm node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("metal tensor: layernorm node %q input shape is required", node.ID)
	}

	dModel := inputShape[len(inputShape)-1]
	weight, err := floatSliceConfigRequired(node, "weight")

	if err != nil {
		return nil, err
	}

	bias, err := floatSliceConfigRequired(node, "bias")

	if err != nil {
		return nil, err
	}

	weightTensor, err := tensorBackend.cachedTensor(node.ID+":weight", []int{dModel}, weight)

	if err != nil {
		return nil, err
	}

	biasTensor, err := tensorBackend.cachedTensor(node.ID+":bias", []int{dModel}, bias)

	if err != nil {
		return nil, err
	}

	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.LayerNormTensor(inputs[0], weightTensor, biasTensor, floatConfig(node, "eps", 1e-5))
}

func (tensorBackend *TensorBackend) applyViewAsHeads(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: view_as_heads node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 3 {
		return nil, fmt.Errorf("metal tensor: view_as_heads node %q expects rank 3", node.ID)
	}

	numHeads := intConfig(node, "num_heads", 0)

	if numHeads <= 0 || inputShape[2]%numHeads != 0 {
		return nil, fmt.Errorf("metal tensor: invalid head count %d", numHeads)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.ViewAsHeadsTensor(
		inputs[0],
		outputShape,
		inputShape[0],
		inputShape[1],
		numHeads,
		inputShape[2]/numHeads,
	)
}

func (tensorBackend *TensorBackend) applyMergeHeads(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: merge_heads node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("metal tensor: merge_heads node %q expects rank 4", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.MergeHeadsTensor(
		inputs[0],
		outputShape,
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
	)
}

func (tensorBackend *TensorBackend) applyLastToken(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: last_token node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) < 2 {
		return nil, fmt.Errorf("metal tensor: last_token node %q expects rank >= 2", node.ID)
	}

	sequenceLength := inputShape[len(inputShape)-2]
	featureLength := inputShape[len(inputShape)-1]
	outerLength := 1

	for _, dimension := range inputShape[:len(inputShape)-2] {
		if dimension <= 0 || outerLength > math.MaxInt/dimension {
			return nil, fmt.Errorf("metal tensor: invalid last_token outer dimensions")
		}

		outerLength *= dimension
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.LastTokenTensor(inputs[0], outputShape, outerLength, sequenceLength, featureLength)
}

func (tensorBackend *TensorBackend) applyLinear(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: linear node %q requires 1 input", node.ID)
	}

	inFeatures := intConfig(node, "in_features", 0)
	outFeatures := intConfig(node, "out_features", 0)

	if inFeatures <= 0 || outFeatures <= 0 {
		return nil, fmt.Errorf("metal tensor: linear node %q requires in/out features", node.ID)
	}

	inputLength := inputs[0].Shape().Len()

	if inputLength%inFeatures != 0 {
		return nil, fmt.Errorf("metal tensor: linear input length %d is not divisible by %d", inputLength, inFeatures)
	}

	weight, err := floatSliceConfigRequired(node, "weight")

	if err != nil {
		return nil, err
	}

	weightTensor, err := tensorBackend.cachedTensor(
		node.ID+":weight",
		[]int{inFeatures, outFeatures},
		weight,
	)

	if err != nil {
		return nil, err
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	bias := floatSliceConfig(node, "bias")
	M := inputLength / inFeatures

	if len(bias) == 0 {
		return mathOps.MatmulFlatTensor(
			inputs[0], weightTensor, outputShape, M, inFeatures, outFeatures,
		)
	}

	biasTensor, err := tensorBackend.cachedTensor(node.ID+":bias", []int{outFeatures}, bias)

	if err != nil {
		return nil, err
	}

	return mathOps.MatmulAddFlatTensor(
		inputs[0], weightTensor, biasTensor, outputShape, M, inFeatures, outFeatures,
	)
}

func (tensorBackend *TensorBackend) applySDPA(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: SDPA node %q requires 3 inputs", node.ID)
	}

	queryShape := inputs[0].Shape().Dims()

	if len(queryShape) != 4 {
		return nil, fmt.Errorf("metal tensor: SDPA node %q query must be rank 4", node.ID)
	}

	keyValueShape := inputs[1].Shape().Dims()

	if len(keyValueShape) != 4 {
		return nil, fmt.Errorf("metal tensor: SDPA node %q key/value must be rank 4", node.ID)
	}

	if !inputs[1].Shape().Equal(inputs[2].Shape()) {
		return nil, fmt.Errorf("metal tensor: SDPA node %q key/value shape mismatch", node.ID)
	}

	if queryShape[0] != keyValueShape[0] || queryShape[1] != keyValueShape[1] ||
		queryShape[3] != keyValueShape[3] {
		return nil, fmt.Errorf("metal tensor: SDPA node %q query/key/value head shape mismatch", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	attentionOps, err := tensorBackend.attention()

	if err != nil {
		return nil, err
	}

	keyTensor := inputs[1]
	valueTensor := inputs[2]
	keyValueLen := keyValueShape[2]
	keyValueStride := keyValueShape[2]
	cache, _ := node.Metadata["kv_cache"].(*kv.Cache)

	if cache != nil {
		cacheEntry, err := tensorBackend.appendResidentKV(
			attentionOps,
			cache,
			node.ID,
			inputs[1],
			inputs[2],
		)

		if err != nil {
			return nil, err
		}

		keyTensor = cacheEntry.key
		valueTensor = cacheEntry.value
		keyValueLen = cacheEntry.shape[2]
		keyValueStride = cacheEntry.capacity
	}

	return attentionOps.SDPATensor(
		inputs[0],
		keyTensor,
		valueTensor,
		outputShape,
		queryShape[0],
		queryShape[1],
		queryShape[2],
		keyValueLen,
		keyValueStride,
		queryShape[3],
		boolConfig(node, "causal", false),
	)
}

func (tensorBackend *TensorBackend) applyGQA(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: GQA node %q requires 3 inputs", node.ID)
	}

	queryShape := inputs[0].Shape().Dims()

	if len(queryShape) != 4 {
		return nil, fmt.Errorf("metal tensor: GQA node %q query must be rank 4", node.ID)
	}

	keyValueShape := inputs[1].Shape().Dims()

	if len(keyValueShape) != 4 {
		return nil, fmt.Errorf("metal tensor: GQA node %q key/value must be rank 4", node.ID)
	}

	if !inputs[1].Shape().Equal(inputs[2].Shape()) {
		return nil, fmt.Errorf("metal tensor: GQA node %q key/value shape mismatch", node.ID)
	}

	numKVHeads := intConfig(node, "num_kv_heads", keyValueShape[1])

	if numKVHeads <= 0 || queryShape[0] != keyValueShape[0] ||
		keyValueShape[1] != numKVHeads ||
		queryShape[3] != keyValueShape[3] || queryShape[1]%numKVHeads != 0 {
		return nil, fmt.Errorf("metal tensor: GQA node %q query/key/value head shape mismatch", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	attentionOps, err := tensorBackend.attention()

	if err != nil {
		return nil, err
	}

	keyTensor := inputs[1]
	valueTensor := inputs[2]
	keyValueLen := keyValueShape[2]
	keyValueStride := keyValueShape[2]
	cache, _ := node.Metadata["kv_cache"].(*kv.Cache)

	if cache != nil {
		if !boolConfig(node, "causal", false) {
			return nil, fmt.Errorf(
				"metal tensor: KV cache is only supported for causal/autoregressive GQA nodes %q "+
					"because it relies on incremental decoding/stateful attention",
				node.ID,
			)
		}

		cacheEntry, err := tensorBackend.appendResidentKV(
			attentionOps,
			cache,
			node.ID,
			inputs[1],
			inputs[2],
		)

		if err != nil {
			return nil, err
		}

		keyTensor = cacheEntry.key
		valueTensor = cacheEntry.value
		keyValueLen = cacheEntry.shape[2]
		keyValueStride = cacheEntry.capacity
	}

	return attentionOps.GQATensor(
		inputs[0],
		keyTensor,
		valueTensor,
		outputShape,
		queryShape[0],
		queryShape[1],
		numKVHeads,
		queryShape[2],
		keyValueLen,
		keyValueStride,
		queryShape[3],
		boolConfig(node, "causal", false),
	)
}

func (tensorBackend *TensorBackend) applyRoPE(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: RoPE node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("metal tensor: RoPE node %q input must be rank 4", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	positionalOps, err := tensorBackend.positional()

	if err != nil {
		return nil, err
	}

	return positionalOps.RoPETensorModeConfig(
		inputs[0],
		outputShape,
		rotary.Config{
			Base:           floatConfig(node, "base", 10000),
			Type:           stringConfig(node, "rope_type", ""),
			Factor:         floatConfig(node, "rope_factor", 0),
			LowFreqFactor:  floatConfig(node, "rope_low_freq_factor", 0),
			HighFreqFactor: floatConfig(node, "rope_high_freq_factor", 0),
			OriginalMaxPositionEmbeddings: intConfig(
				node,
				"rope_original_context",
				0,
			),
		},
		intConfig(node, "position_start", 0),
		stringConfig(node, "mode", ""),
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
	)
}

func (tensorBackend *TensorBackend) appendResidentKV(
	attentionOps *MetalAttention,
	cache *kv.Cache,
	nodeID string,
	keyChunk tensor.Float64Tensor,
	valueChunk tensor.Float64Tensor,
) (*residentKVEntry, error) {
	if cache == nil {
		return nil, fmt.Errorf("metal tensor: KV cache is required")
	}

	chunkShape := keyChunk.Shape().Dims()

	if len(chunkShape) != 4 {
		return nil, fmt.Errorf("metal tensor: KV cache chunk must be rank 4")
	}

	if !keyChunk.Shape().Equal(valueChunk.Shape()) {
		return nil, fmt.Errorf("metal tensor: KV cache key/value chunk shape mismatch")
	}

	if chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0 || chunkShape[3] <= 0 {
		return nil, fmt.Errorf("metal tensor: KV cache chunk dimensions must be positive")
	}

	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	epoch := cache.Epoch()
	entry := tensorBackend.kvEntries[nodeID]

	if entry != nil && entry.epoch != epoch {
		if err := entry.close(); err != nil {
			return nil, err
		}

		delete(tensorBackend.kvEntries, nodeID)
		entry = nil
	}

	previousLen := 0
	capacity := cache.Capacity()

	if entry != nil {
		if err := residentKVShapeAppend(entry.shape, chunkShape); err != nil {
			return nil, err
		}

		previousLen = entry.shape[2]
	}

	requiredLength := previousLen + chunkShape[2]

	if capacity < requiredLength {
		capacity = nextResidentKVCapacity(requiredLength, entry)
	}

	entry, err := tensorBackend.ensureResidentKVCapacity(
		attentionOps,
		nodeID,
		entry,
		epoch,
		chunkShape,
		capacity,
	)

	if err != nil {
		return nil, err
	}

	if err := attentionOps.WriteKVTensor(
		entry.key,
		entry.value,
		keyChunk,
		valueChunk,
		chunkShape[0],
		chunkShape[1],
		previousLen,
		chunkShape[2],
		chunkShape[3],
		entry.capacity,
	); err != nil {
		return nil, err
	}

	entry.shape[2] = requiredLength

	return entry, nil
}

func (tensorBackend *TensorBackend) ensureResidentKVCapacity(
	attentionOps *MetalAttention,
	nodeID string,
	entry *residentKVEntry,
	epoch uint64,
	chunkShape []int,
	capacity int,
) (*residentKVEntry, error) {
	if entry != nil && entry.capacity >= capacity {
		return entry, nil
	}

	outputShapeData := []int{chunkShape[0], chunkShape[1], capacity, chunkShape[3]}
	outputShape, err := tensor.NewShape(outputShapeData)

	if err != nil {
		return nil, err
	}

	key, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	value, err := newMetalTensor(outputShape)

	if err != nil {
		_ = key.Close()

		return nil, err
	}

	currentLength := 0

	if entry != nil {
		currentLength = entry.shape[2]
	}

	if entry != nil && entry.shape[2] > 0 {
		if err := attentionOps.RepackKVTensor(
			entry.key,
			entry.value,
			key,
			value,
			chunkShape[0],
			chunkShape[1],
			entry.shape[2],
			chunkShape[3],
			entry.capacity,
			capacity,
		); err != nil {
			_ = key.Close()
			_ = value.Close()

			return nil, err
		}

		if err := entry.close(); err != nil {
			_ = key.Close()
			_ = value.Close()

			return nil, err
		}
	}

	entry = &residentKVEntry{
		epoch:    epoch,
		capacity: capacity,
		shape:    []int{chunkShape[0], chunkShape[1], currentLength, chunkShape[3]},
		key:      key,
		value:    value,
	}
	tensorBackend.kvEntries[nodeID] = entry

	return entry, nil
}

func nextResidentKVCapacity(requiredLength int, entry *residentKVEntry) int {
	capacity := requiredLength

	if entry != nil && entry.capacity > 0 {
		if entry.capacity > math.MaxInt/2 {
			return requiredLength
		}

		capacity = entry.capacity * 2
	}

	for capacity < requiredLength {
		if capacity > math.MaxInt/2 {
			return requiredLength
		}

		capacity *= 2
	}

	return capacity
}

func residentKVShapeAppend(previousShape []int, chunkShape []int) error {
	if len(previousShape) != 4 || len(chunkShape) != 4 {
		return fmt.Errorf("metal tensor: KV cache shape rank changed")
	}

	for _, dimension := range []int{0, 1, 3} {
		if previousShape[dimension] == chunkShape[dimension] {
			continue
		}

		return fmt.Errorf(
			"metal tensor: KV cache dimension %d changed from %d to %d",
			dimension,
			previousShape[dimension],
			chunkShape[dimension],
		)
	}

	return nil
}

func (tensorBackend *TensorBackend) activation() (*MetalActivation, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.activationOps != nil {
		return tensorBackend.activationOps, nil
	}

	activationOps, err := newMetalActivation(nil)

	if err != nil {
		return nil, err
	}

	tensorBackend.activationOps = activationOps

	return activationOps, nil
}

func (tensorBackend *TensorBackend) math() (*MathOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.mathOps != nil {
		return tensorBackend.mathOps, nil
	}

	mathOps, err := newMetalMath(nil)

	if err != nil {
		return nil, err
	}

	tensorBackend.mathOps = mathOps

	return mathOps, nil
}

func (tensorBackend *TensorBackend) shape() (*MetalShapeOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.shapeOps != nil {
		return tensorBackend.shapeOps, nil
	}

	shapeOps, err := newMetalShape(nil)

	if err != nil {
		return nil, err
	}

	tensorBackend.shapeOps = shapeOps

	return shapeOps, nil
}

func (tensorBackend *TensorBackend) attention() (*MetalAttention, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.attentionOps != nil {
		return tensorBackend.attentionOps, nil
	}

	attentionOps, err := newMetalAttention(nil)

	if err != nil {
		return nil, err
	}

	tensorBackend.attentionOps = attentionOps

	return attentionOps, nil
}

func (tensorBackend *TensorBackend) positional() (*MetalPositional, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.positionalOps != nil {
		return tensorBackend.positionalOps, nil
	}

	positionalOps, err := newMetalPositional(nil)

	if err != nil {
		return nil, err
	}

	tensorBackend.positionalOps = positionalOps

	return positionalOps, nil
}

func (tensorBackend *TensorBackend) embedding(vocabSize, dModel int) (*EmbeddingOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	key := fmt.Sprintf("%d:%d", vocabSize, dModel)

	if embeddingOps := tensorBackend.embeddingOps[key]; embeddingOps != nil {
		return embeddingOps, nil
	}

	embeddingOps, err := NewEmbeddingOps(metalLibrary(nil, "embedding.metallib"), vocabSize, dModel)

	if err != nil {
		return nil, err
	}

	tensorBackend.embeddingOps[key] = embeddingOps

	return embeddingOps, nil
}

func (tensorBackend *TensorBackend) cachedTensor(
	key string,
	shapeData []int,
	values []float64,
) (tensor.Float64Tensor, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("metal tensor: cached tensor %q has no values", key)
	}

	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if value := tensorBackend.resident[key]; value != nil {
		return value, nil
	}

	shape, err := tensor.NewShape(shapeData)

	if err != nil {
		return nil, err
	}

	value, err := tensorBackend.UploadFloat64(shape, values)

	if err != nil {
		return nil, err
	}

	tensorBackend.resident[key] = value

	return value, nil
}

func floatSliceConfigRequired(node executor.NodeSpec, key string) ([]float64, error) {
	values := floatSliceConfig(node, key)

	if len(values) == 0 {
		return nil, fmt.Errorf("metal tensor: node %q requires %s", node.ID, key)
	}

	return values, nil
}

func floatSliceConfig(node executor.NodeSpec, key string) []float64 {
	value, ok := node.Metadata[key]

	if !ok {
		return nil
	}

	switch typed := value.(type) {
	case []float64:
		return typed
	case []float32:
		values := make([]float64, len(typed))

		for index, value := range typed {
			values[index] = float64(value)
		}

		return values
	default:
		return nil
	}
}

func intConfig(node executor.NodeSpec, key string, fallback int) int {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return fallback
	}
}

func stringConfig(node executor.NodeSpec, key string, fallback string) string {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	typed, ok := value.(string)

	if !ok {
		return fallback
	}

	return typed
}

func floatConfig(node executor.NodeSpec, key string, fallback float64) float64 {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return fallback
	}
}

func boolConfig(node executor.NodeSpec, key string, fallback bool) bool {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	typed, ok := value.(bool)

	if !ok {
		return fallback
	}

	return typed
}
