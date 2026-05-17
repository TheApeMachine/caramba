//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
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
	case "math.exp":
		return tensorBackend.applyExp(ctx, node, inputs)
	case "math.log":
		return tensorBackend.applyLog(ctx, node, inputs)
	case "math.sign":
		return tensorBackend.applySign(ctx, node, inputs)
	case "math.outer":
		return tensorBackend.applyOuter(ctx, node, inputs)
	case "math.inv_sqrt_dim_scale":
		return tensorBackend.applyInvSqrtDimScale(ctx, node, inputs)
	case "math.dropout":
		return tensorBackend.applyDropout(ctx, node, inputs)
	case "math.softmax":
		return tensorBackend.applySoftmax(ctx, node, inputs)
	case "math.logsumexp":
		return tensorBackend.applyLogSumExp(ctx, node, inputs)
	case "math.rmsnorm":
		return tensorBackend.applyRMSNorm(ctx, node, inputs)
	case "math.layernorm":
		return tensorBackend.applyLayerNorm(ctx, node, inputs)
	case "math.groupnorm":
		return tensorBackend.applyGroupNorm(ctx, node, inputs)
	case "shape.upsample_nearest2d":
		return tensorBackend.applyUpsampleNearest2D(ctx, node, inputs)
	case "shape.reshape":
		return tensorBackend.applyReshape(ctx, node, inputs)
	case "shape.transpose":
		return tensorBackend.applyTranspose(ctx, node, inputs)
	case "shape.concat":
		return tensorBackend.applyConcat(ctx, node, inputs)
	case "shape.split":
		return tensorBackend.applySplit(ctx, node, inputs)
	case "shape.view_as_heads":
		return tensorBackend.applyViewAsHeads(ctx, node, inputs)
	case "shape.merge_heads":
		return tensorBackend.applyMergeHeads(ctx, node, inputs)
	case "shape.last_token":
		return tensorBackend.applyLastToken(ctx, node, inputs)
	case "projection.linear":
		return tensorBackend.applyLinear(ctx, node, inputs)
	case "projection.fused_qkv":
		return tensorBackend.applyFusedQKV(ctx, node, inputs)
	case "attention.sdpa":
		return tensorBackend.applySDPA(ctx, node, inputs)
	case "attention.mqa":
		return tensorBackend.applyMQA(ctx, node, inputs)
	case "attention.gqa":
		return tensorBackend.applyGQA(ctx, node, inputs)
	case "attention.sliding_window":
		return tensorBackend.applySlidingWindowAttention(ctx, node, inputs)
	case "positional.rope":
		return tensorBackend.applyRoPE(ctx, node, inputs)
	case "positional.alibi":
		return tensorBackend.applyALiBi(ctx, node, inputs)
	case "convolution.conv1d":
		return tensorBackend.applyConv1D(ctx, node, inputs)
	case "convolution.conv2d":
		return tensorBackend.applyConv2D(ctx, node, inputs)
	case "convolution.conv3d":
		return tensorBackend.applyConv3D(ctx, node, inputs)
	case "convolution.conv_transpose2d":
		return tensorBackend.applyConvTranspose2D(ctx, node, inputs)
	case "pooling.max_pool2d":
		return tensorBackend.applyMaxPool2D(ctx, node, inputs)
	case "pooling.avg_pool2d":
		return tensorBackend.applyAvgPool2D(ctx, node, inputs)
	case "pooling.adaptive_avg_pool2d":
		return tensorBackend.applyAdaptiveAvgPool2D(ctx, node, inputs)
	case "pooling.adaptive_max_pool2d":
		return tensorBackend.applyAdaptiveMaxPool2D(ctx, node, inputs)
	case "masking.apply":
		return tensorBackend.applyMask(ctx, node, inputs)
	case "masking.causal":
		return tensorBackend.applyCausalMask(ctx, node, inputs)
	case "vsa.bind":
		return tensorBackend.applyVSABind(ctx, node, inputs)
	case "vsa.bundle":
		return tensorBackend.applyVSABundle(ctx, node, inputs)
	case "vsa.similarity":
		return tensorBackend.applyVSASimilarity(ctx, node, inputs)
	case "vsa.permute":
		return tensorBackend.applyVSAPermute(ctx, node, inputs)
	case "vsa.inverse_permute":
		return tensorBackend.applyVSAInversePermute(ctx, node, inputs)
	case "hawkes.intensity":
		return tensorBackend.applyHawkesIntensity(ctx, node, inputs)
	case "hawkes.kernel_matrix":
		return tensorBackend.applyHawkesKernelMatrix(ctx, node, inputs)
	case "hawkes.log_likelihood":
		return tensorBackend.applyHawkesLogLikelihood(ctx, node, inputs)
	case "hawkes.simulate":
		return tensorBackend.applyHawkesSimulate(ctx, node, inputs)
	case "active_inference.free_energy":
		return tensorBackend.applyActiveFreeEnergy(ctx, node, inputs)
	case "active_inference.belief_update":
		return tensorBackend.applyActiveBeliefUpdate(ctx, node, inputs)
	case "active_inference.precision_weight":
		return tensorBackend.applyActivePrecisionWeight(ctx, node, inputs)
	case "active_inference.expected_free_energy":
		return tensorBackend.applyActiveExpectedFreeEnergy(ctx, node, inputs)
	case "predictive_coding.prediction":
		return tensorBackend.applyPredictivePrediction(ctx, node, inputs)
	case "predictive_coding.prediction_error":
		return tensorBackend.applyPredictivePredictionError(ctx, node, inputs)
	case "predictive_coding.update_representation":
		return tensorBackend.applyPredictiveUpdateRepresentation(ctx, node, inputs)
	case "predictive_coding.update_weights":
		return tensorBackend.applyPredictiveUpdateWeights(ctx, node, inputs)
	case "markov_blanket.partition":
		return tensorBackend.applyMarkovPartition(ctx, node, inputs)
	case "markov_blanket.flow_internal":
		return tensorBackend.applyMarkovFlowInternal(ctx, node, inputs)
	case "markov_blanket.flow_active":
		return tensorBackend.applyMarkovFlowActive(ctx, node, inputs)
	case "markov_blanket.mutual_information":
		return tensorBackend.applyMarkovMutualInformation(ctx, node, inputs)
	case "causal.counterfactual":
		return tensorBackend.applyCausalCounterfactual(ctx, node, inputs)
	case "causal.frontdoor_adjustment":
		return tensorBackend.applyCausalFrontdoorAdjustment(ctx, node, inputs)
	case "causal.backdoor_adjustment":
		return tensorBackend.applyCausalBackdoorAdjustment(ctx, node, inputs)
	case "causal.cate":
		return tensorBackend.applyCausalCATE(ctx, node, inputs)
	case "causal.iv_estimate":
		return tensorBackend.applyCausalIVEstimate(ctx, node, inputs)
	case "causal.dag_markov_factorization":
		return tensorBackend.applyCausalDAGMarkovFactorization(ctx, node, inputs)
	case "causal.do_calculus":
		return tensorBackend.applyCausalDoCalculus(ctx, node, inputs)
	case "train.loss.mse":
		return tensorBackend.applyMSELoss(ctx, node, inputs)
	case "train.loss.cross_entropy":
		return tensorBackend.applyCrossEntropyLoss(ctx, node, inputs)
	case "train.loss.mse_grad", "train.grad.mse":
		return tensorBackend.applyMSEGrad(ctx, node, inputs)
	case "train.loss.cross_entropy_grad", "train.grad.cross_entropy":
		return tensorBackend.applyCrossEntropyGrad(ctx, node, inputs)
	case "bench.accuracy", "bench.metric.accuracy":
		return tensorBackend.applyAccuracy(ctx, node, inputs)
	case "bench.perplexity", "bench.metric.perplexity":
		return tensorBackend.applyPerplexity(ctx, node, inputs)
	case "bench.f1", "bench.metric.f1":
		return tensorBackend.applyF1(ctx, node, inputs)
	case "train.optimizer.adam", "optimizer.adam",
		"train.optimizer.adamw", "optimizer.adamw",
		"train.optimizer.adamax", "optimizer.adamax",
		"train.optimizer.sgd", "optimizer.sgd",
		"train.optimizer.lion", "optimizer.lion",
		"train.optimizer.rmsprop", "optimizer.rmsprop",
		"train.optimizer.hebbian", "optimizer.hebbian",
		"train.optimizer.lars", "optimizer.lars",
		"train.optimizer.lamb", "optimizer.lamb",
		"train.optimizer.adagrad", "optimizer.adagrad",
		"train.optimizer.adadelta", "optimizer.adadelta",
		"train.optimizer.lbfgs", "optimizer.lbfgs":
		return tensorBackend.applyOptimizerStep(ctx, node, inputs)
	default:
		return nil, fmt.Errorf(
			"metal tensor: operation %q node %q has no resident Metal implementation",
			node.Op,
			node.ID,
		)
	}
}

func (tensorBackend *TensorBackend) applyReshape(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: reshape node %q requires 1 input", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.CopyTensor(inputs[0], outputShape)
}

func (tensorBackend *TensorBackend) applyTranspose(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: transpose node %q requires 1 input", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.TransposeTensor(
		inputs[0],
		outputShape,
		intConfig(node, "dim0", 0),
		intConfig(node, "dim1", 1),
	)
}

func (tensorBackend *TensorBackend) applyConcat(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) < 2 {
		return nil, fmt.Errorf("metal tensor: concat node %q requires at least 2 inputs", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	if len(inputs) == 2 {
		return shapeOps.ConcatTensor(inputs[0], inputs[1], outputShape)
	}

	current := inputs[0]
	var temporary tensor.Float64Tensor

	for inputIndex := 1; inputIndex < len(inputs); inputIndex++ {
		nextLength := current.Shape().Len() + inputs[inputIndex].Shape().Len()
		nextShape := outputShape

		if inputIndex != len(inputs)-1 {
			nextShape, err = tensor.NewShape([]int{nextLength})

			if err != nil {
				if temporary != nil {
					_ = temporary.Close()
				}

				return nil, err
			}
		}

		next, err := shapeOps.ConcatTensor(current, inputs[inputIndex], nextShape)

		if temporary != nil {
			_ = temporary.Close()
		}

		if err != nil {
			return nil, err
		}

		current = next
		temporary = next
	}

	return current, nil
}

func (tensorBackend *TensorBackend) applySplit(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: split node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()
	dimension := intConfig(node, "dim", 0)

	if dimension < 0 || dimension >= len(inputShape) {
		return nil, fmt.Errorf("metal tensor: split node %q dimension out of range", node.ID)
	}

	outer, inner, err := metalSplitOuterInner(inputShape, dimension)

	if err != nil {
		return nil, err
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.SplitTensor(
		inputs[0],
		outputShape,
		outer,
		inputShape[dimension],
		intConfig(node, "split_size", inputShape[dimension]),
		inner,
	)
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

func (tensorBackend *TensorBackend) applyGroupNorm(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: groupnorm node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("metal tensor: groupnorm node %q expects NCHW rank 4", node.ID)
	}

	channels := inputShape[1]
	groups := intConfig(node, "groups", 0)
	groups = intConfig(node, "num_groups", groups)

	weight, err := floatSliceConfigRequired(node, "weight")

	if err != nil {
		return nil, err
	}

	bias, err := floatSliceConfigRequired(node, "bias")

	if err != nil {
		return nil, err
	}

	weightTensor, err := tensorBackend.cachedTensor(node.ID+":weight", []int{channels}, weight)

	if err != nil {
		return nil, err
	}

	biasTensor, err := tensorBackend.cachedTensor(node.ID+":bias", []int{channels}, bias)

	if err != nil {
		return nil, err
	}

	mathOps, err := tensorBackend.math()

	if err != nil {
		return nil, err
	}

	return mathOps.GroupNormTensor(
		inputs[0],
		weightTensor,
		biasTensor,
		groups,
		floatConfig(node, "eps", 1e-5),
	)
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

func (tensorBackend *TensorBackend) applyUpsampleNearest2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: upsample_nearest2d node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("metal tensor: upsample_nearest2d node %q expects rank 4", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	stateDict := executor.OperationConfig(node)
	stateDict.OutH = outputShape.Dims()[2]
	stateDict.OutW = outputShape.Dims()[3]
	scaleH, scaleW, err := metalUpsampleNearest2DScale(stateDict, inputShape)

	if err != nil {
		return nil, err
	}

	shapeOps, err := tensorBackend.shape()

	if err != nil {
		return nil, err
	}

	return shapeOps.UpsampleNearest2DTensor(
		inputs[0],
		outputShape,
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
		scaleH,
		scaleW,
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

func (tensorBackend *TensorBackend) applyConv2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: conv2d node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("metal tensor: conv2d node %q expects NCHW rank 4", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	outputDims := outputShape.Dims()

	if len(outputDims) != 4 {
		return nil, fmt.Errorf("metal tensor: conv2d node %q output must be rank 4", node.ID)
	}

	outChannels := intConfigAny(node, 0, "out_channels", "out_c")
	kernelHeight := intConfigAny(node, intConfig(node, "kernel_size", 0), "kernel_h", "k_h")
	kernelWidth := intConfigAny(node, intConfig(node, "kernel_size", 0), "kernel_w", "k_w")
	stride := intConfig(node, "stride", 1)
	strideHeight := intConfigAny(node, stride, "stride_h", "s_h")
	strideWidth := intConfigAny(node, stride, "stride_w", "s_w")
	padding := intConfig(node, "padding", 0)
	padHeight := intConfigAny(node, padding, "pad_h", "p_h")
	padWidth := intConfigAny(node, padding, "pad_w", "p_w")
	dilation := intConfig(node, "dilation", 1)
	dilationHeight := intConfigAny(node, dilation, "dilation_h", "dil_h", "d_h")
	dilationWidth := intConfigAny(node, dilation, "dilation_w", "dil_w", "d_w")
	groups := intConfig(node, "groups", 1)
	groups = intConfig(node, "num_groups", groups)

	if outChannels == 0 {
		outChannels = outputDims[1]
	}

	if err := validateMetalConvNode(
		node.ID,
		"conv2d",
		inputShape[1],
		outputDims[1],
		outChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		dilationHeight,
		dilationWidth,
		groups,
	); err != nil {
		return nil, err
	}

	weightTensor, biasTensor, err := tensorBackend.convolutionTensors(
		node,
		"conv2d",
		[]int{outChannels, inputShape[1] / groups, kernelHeight, kernelWidth},
		[]int{outChannels},
	)

	if err != nil {
		return nil, err
	}

	convolutionOps, err := tensorBackend.convolution()

	if err != nil {
		return nil, err
	}

	return convolutionOps.Conv2dTensor(
		inputs[0],
		weightTensor,
		biasTensor,
		outputShape,
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
		outChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		padHeight,
		padWidth,
		dilationHeight,
		dilationWidth,
		groups,
	)
}

func (tensorBackend *TensorBackend) applyConvTranspose2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: conv_transpose2d node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf(
			"metal tensor: conv_transpose2d node %q expects NCHW rank 4",
			node.ID,
		)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	outputDims := outputShape.Dims()

	if len(outputDims) != 4 {
		return nil, fmt.Errorf(
			"metal tensor: conv_transpose2d node %q output must be rank 4",
			node.ID,
		)
	}

	outChannels := intConfigAny(node, outputDims[1], "out_channels", "out_c")
	kernelHeight := intConfigAny(node, intConfig(node, "kernel_size", 0), "kernel_h", "k_h")
	kernelWidth := intConfigAny(node, intConfig(node, "kernel_size", 0), "kernel_w", "k_w")
	stride := intConfig(node, "stride", 1)
	strideHeight := intConfigAny(node, stride, "stride_h", "s_h")
	strideWidth := intConfigAny(node, stride, "stride_w", "s_w")
	padding := intConfig(node, "padding", 0)
	padHeight := intConfigAny(node, padding, "pad_h", "p_h")
	padWidth := intConfigAny(node, padding, "pad_w", "p_w")
	dilation := intConfig(node, "dilation", 1)
	dilationHeight := intConfigAny(node, dilation, "dilation_h", "dil_h", "d_h")
	dilationWidth := intConfigAny(node, dilation, "dilation_w", "dil_w", "d_w")
	groups := intConfig(node, "groups", 1)
	groups = intConfig(node, "num_groups", groups)
	outPadHeight := intConfig(node, "out_pad_h", 0)
	outPadWidth := intConfig(node, "out_pad_w", 0)

	if err := validateMetalConvNode(
		node.ID,
		"conv_transpose2d",
		inputShape[1],
		outputDims[1],
		outChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		dilationHeight,
		dilationWidth,
		groups,
	); err != nil {
		return nil, err
	}

	weightTensor, biasTensor, err := tensorBackend.convolutionTensors(
		node,
		"conv_transpose2d",
		[]int{inputShape[1], outChannels / groups, kernelHeight, kernelWidth},
		[]int{outChannels},
	)

	if err != nil {
		return nil, err
	}

	convolutionOps, err := tensorBackend.convolution()

	if err != nil {
		return nil, err
	}

	return convolutionOps.ConvTranspose2dTensor(
		inputs[0],
		weightTensor,
		biasTensor,
		outputShape,
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
		outChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		padHeight,
		padWidth,
		dilationHeight,
		dilationWidth,
		groups,
		outPadHeight,
		outPadWidth,
	)
}

func validateMetalConvNode(
	nodeID string,
	operation string,
	inputChannels int,
	outputChannelsFromShape int,
	outputChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
) error {
	if outputChannels != outputChannelsFromShape {
		return fmt.Errorf(
			"metal tensor: %s node %q out_channels=%d does not match output shape channels=%d",
			operation,
			nodeID,
			outputChannels,
			outputChannelsFromShape,
		)
	}

	if inputChannels <= 0 || outputChannels <= 0 || kernelHeight <= 0 ||
		kernelWidth <= 0 || strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("metal tensor: %s node %q has invalid dimensions", operation, nodeID)
	}

	if inputChannels%groups == 0 && outputChannels%groups == 0 {
		return nil
	}

	return fmt.Errorf(
		"metal tensor: %s node %q groups=%d must divide InC=%d and OutC=%d",
		operation,
		nodeID,
		groups,
		inputChannels,
		outputChannels,
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

func (tensorBackend *TensorBackend) applyALiBi(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 0 {
		return nil, fmt.Errorf("metal tensor: ALiBi node %q requires 0 inputs", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	positionalOps, err := tensorBackend.positional()

	if err != nil {
		return nil, err
	}

	return positionalOps.ALiBiTensor(outputShape, boolConfig(node, "causal", false))
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

	key, err := tensorBackend.runtime.NewFloat32Tensor(
		outputShape,
		MetalAllocationKVCache,
	)

	if err != nil {
		return nil, err
	}

	value, err := tensorBackend.runtime.NewFloat32Tensor(
		outputShape,
		MetalAllocationKVCache,
	)

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

	activationOps.runtime = tensorBackend.runtime
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

	mathOps.runtime = tensorBackend.runtime
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

	shapeOps.runtime = tensorBackend.runtime
	tensorBackend.shapeOps = shapeOps

	return shapeOps, nil
}

func (tensorBackend *TensorBackend) convolution() (*ConvolutionOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.convolutionOps != nil {
		return tensorBackend.convolutionOps, nil
	}

	convolutionOps, err := NewConvolutionOps(metalLibrary(nil, "convolution.metallib"))

	if err != nil {
		return nil, err
	}

	convolutionOps.runtime = tensorBackend.runtime
	tensorBackend.convolutionOps = convolutionOps

	return convolutionOps, nil
}

func (tensorBackend *TensorBackend) pooling() (*PoolingOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.poolingOps != nil {
		return tensorBackend.poolingOps, nil
	}

	poolingOps, err := NewPoolingOps(metalLibrary(nil, "pooling.metallib"))

	if err != nil {
		return nil, err
	}

	poolingOps.runtime = tensorBackend.runtime
	tensorBackend.poolingOps = poolingOps

	return poolingOps, nil
}

func (tensorBackend *TensorBackend) masking() (*MetalMasking, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.maskingOps != nil {
		return tensorBackend.maskingOps, nil
	}

	maskingOps, err := NewMasking(metalLibrary(nil, "masking.metallib"))

	if err != nil {
		return nil, err
	}

	maskingOps.runtime = tensorBackend.runtime
	tensorBackend.maskingOps = maskingOps

	return maskingOps, nil
}

func (tensorBackend *TensorBackend) projection() (*ProjectionOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.projectionOps != nil {
		return tensorBackend.projectionOps, nil
	}

	projectionOps, err := NewProjectionOps(metalLibrary(nil, "projection.metallib"))

	if err != nil {
		return nil, err
	}

	projectionOps.runtime = tensorBackend.runtime
	tensorBackend.projectionOps = projectionOps

	return projectionOps, nil
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

	attentionOps.runtime = tensorBackend.runtime
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

	positionalOps.runtime = tensorBackend.runtime
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

	embeddingOps.runtime = tensorBackend.runtime
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

func (tensorBackend *TensorBackend) convolutionTensors(
	node executor.NodeSpec,
	name string,
	weightShape []int,
	biasShape []int,
) (tensor.Float64Tensor, tensor.Float64Tensor, error) {
	weight, err := floatSliceConfigRequired(node, "weight")

	if err != nil {
		return nil, nil, err
	}

	bias, err := floatSliceConfigRequired(node, "bias")

	if err != nil {
		return nil, nil, err
	}

	weightTensor, err := tensorBackend.cachedTensor(
		node.ID+":"+name+":weight",
		weightShape,
		weight,
	)

	if err != nil {
		return nil, nil, err
	}

	biasTensor, err := tensorBackend.cachedTensor(
		node.ID+":"+name+":bias",
		biasShape,
		bias,
	)

	if err != nil {
		return nil, nil, err
	}

	return weightTensor, biasTensor, nil
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

func metalUpsampleNearest2DScale(stateDict *state.Dict, shape []int) (int, int, error) {
	if len(shape) != 4 {
		return 0, 0, fmt.Errorf("metal.shape.upsample_nearest2d: expected NCHW rank 4")
	}

	scaleH := stateDict.ScaleH
	scaleW := stateDict.ScaleW

	if scaleH == 0 && stateDict.OutH > 0 {
		if stateDict.OutH%shape[2] != 0 {
			return 0, 0, fmt.Errorf(
				"metal.shape.upsample_nearest2d: out_h %d is not divisible by height %d",
				stateDict.OutH,
				shape[2],
			)
		}

		scaleH = stateDict.OutH / shape[2]
	}

	if scaleW == 0 && stateDict.OutW > 0 {
		if stateDict.OutW%shape[3] != 0 {
			return 0, 0, fmt.Errorf(
				"metal.shape.upsample_nearest2d: out_w %d is not divisible by width %d",
				stateDict.OutW,
				shape[3],
			)
		}

		scaleW = stateDict.OutW / shape[3]
	}

	if scaleH <= 0 || scaleW <= 0 {
		return 0, 0, fmt.Errorf(
			"metal.shape.upsample_nearest2d: scale_h and scale_w must be positive",
		)
	}

	return scaleH, scaleW, nil
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

func intConfigAny(node executor.NodeSpec, fallback int, keys ...string) int {
	for _, key := range keys {
		value := intConfig(node, key, fallback)

		if value != fallback {
			return value
		}
	}

	return fallback
}

func metalSplitOuterInner(shape []int, dimension int) (int, int, error) {
	outer := 1

	for _, value := range shape[:dimension] {
		if value <= 0 {
			return 0, 0, fmt.Errorf("metal tensor: split outer dimensions must be positive")
		}

		outer *= value
	}

	inner := 1

	for _, value := range shape[dimension+1:] {
		if value <= 0 {
			return 0, 0, fmt.Errorf("metal tensor: split inner dimensions must be positive")
		}

		inner *= value
	}

	return outer, inner, nil
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
