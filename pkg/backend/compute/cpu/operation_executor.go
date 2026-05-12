package cpu

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/bench"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/convolution"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/hawkes"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/markov_blanket"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/masking"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/pooling"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/positional"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/predictive_coding"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/projection"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/shape"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/train"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/vsa"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

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
		return nil, fmt.Errorf("cpu tensor: input node %q was not materialized", node.ID)
	case ir.OpAdd:
		return requireInputs(node, inputs, 2, tensorBackend.Add)
	case ir.OpMul:
		return requireInputs(node, inputs, 2, tensorBackend.Mul)
	case ir.OpMatmul:
		return requireInputs(node, inputs, 2, tensorBackend.Matmul)
	case ir.OpReLU:
		return requireInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.ReLU(input)
		})
	case ir.OpLeakyReLU:
		return requireInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.LeakyReLU(input, floatConfig(node, "alpha", 0.01))
		})
	case ir.OpGELU:
		return requireInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.GELU(input)
		})
	case ir.OpTanh:
		return requireInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Tanh(input)
		})
	case ir.OpSigmoid:
		return requireInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Sigmoid(input)
		})
	case ir.OpSwiGLU:
		return requireInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.SwiGLU(input)
		})
	case ir.OpFused:
		return tensorBackend.applyFused(ctx, node, inputs)
	default:
		return tensorBackend.applyOperation(ctx, node, inputs)
	}
}

func (tensorBackend *TensorBackend) applyFused(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	activation, _ := node.Metadata["activation"].(string)
	if len(inputs) == 2 {
		output, err := tensorBackend.Matmul(inputs[0], inputs[1])

		if err != nil {
			return nil, err
		}

		switch {
		case strings.EqualFold(activation, string(ir.OpReLU)):
			return tensorBackend.ReLU(output)
		case strings.EqualFold(activation, string(ir.OpGELU)):
			return tensorBackend.GELU(output)
		default:
			return output, nil
		}
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("cpu tensor: Fused node %q requires 2 or 3 inputs", node.ID)
	}

	if strings.EqualFold(activation, string(ir.OpGELU)) {
		return tensorBackend.MatmulAddGELU(inputs[0], inputs[1], inputs[2])
	}

	return tensorBackend.MatmulAdd(inputs[0], inputs[1], inputs[2])
}

func (tensorBackend *TensorBackend) applyOperation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	switch strings.ToLower(string(node.Op)) {
	case "activation.swish":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewMul())
	case "attention.sdpa":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, attention.NewSDPA())
	case "attention.mqa":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, attention.NewMQA())
	case "attention.gqa":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, attention.NewGQA())
	case "attention.sliding_window":
		return executor.RunOperation(
			ctx,
			tensorBackend,
			node,
			inputs,
			attention.NewSlidingWindow(intConfig(node, "window", 1)),
		)
	case "masking.apply":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, masking.NewApplyMask())
	case "masking.causal":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, masking.NewCausalMask())
	case "math.exp":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewExp())
	case "math.log":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewLog())
	case "math.logsumexp":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewLogSumExp())
	case "math.softmax":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewSoftmax())
	case "math.outer":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewOuter())
	case "math.sign":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewSign())
	case "math.inv_sqrt_dim_scale":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewInvSqrtDimScale())
	case "math.dropout":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewDropout(
			floatConfig(node, "p", 0),
			boolConfig(node, "training", false),
		))
	case "math.rmsnorm":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewRMSNorm(
			floatConfig(node, "eps", 1e-5),
			floatSliceConfig(node, "weight"),
		))
	case "math.layernorm":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewLayerNorm(
			floatConfig(node, "eps", 1e-5),
			floatSliceConfig(node, "weight"),
			floatSliceConfig(node, "bias"),
		))
	case "shape.reshape":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, shape.NewReshape(intSliceConfig(node, "shape")))
	case "shape.transpose":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, shape.NewTranspose(
			intConfig(node, "dim0", 0),
			intConfig(node, "dim1", 1),
		))
	case "shape.concat":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, shape.NewConcat(intConfig(node, "dim", 0)))
	case "shape.split":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, shape.NewSplit(
			intConfig(node, "split_size", 1),
			intConfig(node, "dim", 0),
		))
	case "shape.view_as_heads":
		return executor.RunOperation(
			ctx,
			tensorBackend,
			node,
			inputs,
			shape.NewViewAsHeads(intConfig(node, "num_heads", 1)),
		)
	case "shape.merge_heads":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, shape.NewMergeHeads())
	case "positional.rope":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, positional.NewRoPE(
			floatConfig(node, "base", 10000),
			intConfig(node, "head_dim", 1),
		))
	case "positional.alibi":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, positional.NewALiBi(
			intConfig(node, "num_heads", 1),
			boolConfig(node, "causal", true),
		))
	case "convolution.conv1d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, convolution.NewConv1d(
			intConfig(node, "in_c", 1),
			intConfig(node, "out_c", 1),
			intConfig(node, "kernel_size", 1),
			intConfig(node, "stride", 1),
			intConfig(node, "padding", 0),
			intConfig(node, "dilation", 1),
			intConfig(node, "groups", 1),
		))
	case "convolution.conv2d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, convolution.NewConv2d(
			intConfig(node, "in_c", 1), intConfig(node, "out_c", 1),
			intConfig(node, "k_h", 1), intConfig(node, "k_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			intConfig(node, "groups", 1),
		))
	case "convolution.conv3d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, convolution.NewConv3d(
			intConfig(node, "in_c", 1), intConfig(node, "out_c", 1),
			intConfig(node, "k_d", 1), intConfig(node, "k_h", 1), intConfig(node, "k_w", 1),
			intConfig(node, "s_d", 1), intConfig(node, "s_h", 1), intConfig(node, "s_w", 1),
			intConfig(node, "p_d", 0), intConfig(node, "p_h", 0), intConfig(node, "p_w", 0),
			intConfig(node, "dil_d", 1), intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			intConfig(node, "groups", 1),
		))
	case "convolution.conv_transpose2d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, convolution.NewConvTranspose2d(
			intConfig(node, "in_c", 1), intConfig(node, "out_c", 1),
			intConfig(node, "k_h", 1), intConfig(node, "k_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "out_pad_h", 0), intConfig(node, "out_pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			intConfig(node, "groups", 1),
		))
	case "pooling.max_pool2d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, pooling.NewMaxPool2d(
			intConfig(node, "kernel_h", 1), intConfig(node, "kernel_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			boolConfig(node, "ceil", false),
		))
	case "pooling.avg_pool2d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, pooling.NewAvgPool2d(
			intConfig(node, "kernel_h", 1), intConfig(node, "kernel_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			boolConfig(node, "ceil", false), boolConfig(node, "count_include_pad", false),
			intConfig(node, "divisor_override", 0),
		))
	case "pooling.adaptive_avg_pool2d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, pooling.NewAdaptiveAvgPool2d(
			intConfig(node, "out_h", 1),
			intConfig(node, "out_w", 1),
		))
	case "pooling.adaptive_max_pool2d":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, pooling.NewAdaptiveMaxPool2d(
			intConfig(node, "out_h", 1),
			intConfig(node, "out_w", 1),
		))
	case "projection.linear":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, projection.NewLinear(
			intConfig(node, "in_features", 1),
			intConfig(node, "out_features", 1),
		))
	case "projection.fused_qkv":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, projection.NewFusedQKV(
			intConfig(node, "d_in", 1), intConfig(node, "d_q", 1),
			intConfig(node, "d_k", 1), intConfig(node, "d_v", 1),
		))
	case "hawkes.intensity":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, hawkes.NewIntensity())
	case "hawkes.kernel_matrix":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, hawkes.NewKernelMatrix())
	case "hawkes.log_likelihood":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, hawkes.NewLogLikelihood())
	case "hawkes.simulate":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, hawkes.NewSimulate())
	case "vsa.bind":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, vsa.NewBind())
	case "vsa.bundle":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, vsa.NewBundle())
	case "vsa.similarity":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, vsa.NewSimilarity())
	case "vsa.permute":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, vsa.NewPermute(intConfig(node, "k", 1)))
	case "vsa.inverse_permute":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, vsa.NewInversePermute(intConfig(node, "k", 1)))
	case "active_inference.belief_update":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, active_inference.NewBeliefUpdate())
	case "active_inference.expected_free_energy":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, active_inference.NewExpectedFreeEnergy())
	case "active_inference.free_energy":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, active_inference.NewFreeEnergy())
	case "active_inference.precision_weight":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, active_inference.NewPrecisionWeight())
	case "predictive_coding.prediction":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, predictive_coding.NewPrediction())
	case "predictive_coding.prediction_error":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, predictive_coding.NewPredictionError())
	case "predictive_coding.update_representation":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, predictive_coding.NewUpdateRepresentation())
	case "predictive_coding.update_weights":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, predictive_coding.NewUpdateWeights())
	case "markov_blanket.flow_active":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, markov_blanket.NewFlowActive())
	case "markov_blanket.flow_internal":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, markov_blanket.NewFlowInternal())
	case "markov_blanket.mutual_information":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, markov_blanket.NewMutualInformation())
	case "markov_blanket.partition":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, markov_blanket.NewPartition())
	case "causal.backdoor_adjustment":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewBackdoorAdjustment())
	case "causal.cate":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewCATE())
	case "causal.counterfactual":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewCounterfactual())
	case "causal.dag_markov_factorization":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewDAGMarkovFactorization())
	case "causal.do_calculus":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewDoCalculus())
	case "causal.frontdoor_adjustment":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewFrontdoorAdjustment())
	case "causal.iv_estimate":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, causal.NewIVEstimate())
	case "train.loss.mse":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewMSELoss())
	case "train.loss.cross_entropy":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewCrossEntropyLoss())
	case "train.loss.mse_grad":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewMSEGrad())
	case "train.loss.cross_entropy_grad":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewCrossEntropyGrad())
	case "train.optimizer.adam":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewAdamStep(
			floatConfig(node, "lr", 1e-3), floatConfig(node, "beta1", 0.9),
			floatConfig(node, "beta2", 0.999), floatConfig(node, "eps", 1e-8),
			floatConfig(node, "wd", 0),
		))
	case "train.optimizer.adamw":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewAdamWStep(
			floatConfig(node, "lr", 1e-3), floatConfig(node, "beta1", 0.9),
			floatConfig(node, "beta2", 0.999), floatConfig(node, "eps", 1e-8),
			floatConfig(node, "wd", 0),
		))
	case "train.optimizer.sgd":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewSGDStep(
			floatConfig(node, "lr", 1e-3), floatConfig(node, "momentum", 0),
			floatConfig(node, "wd", 0), boolConfig(node, "nesterov", false),
		))
	case "bench.accuracy":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, bench.NewAccuracy())
	case "bench.perplexity":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, bench.NewPerplexity())
	case "bench.f1":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, bench.NewF1())
	case "model.graft":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, model.NewGraft(
			stringConfig(node, "source", ""),
			stringConfig(node, "at", ""),
			stringConfig(node, "mode", "append"),
		))
	case "model.freeze":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, model.NewFreeze(
			stringConfig(node, "source", ""),
			stringConfig(node, "pattern", ""),
			stringConfig(node, "except", ""),
			boolConfig(node, "frozen", true),
		))
	default:
		return nil, fmt.Errorf("cpu tensor: unsupported operation %q", node.Op)
	}
}

func requireInputs(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
	count int,
	apply func(tensor.Float64Tensor, tensor.Float64Tensor) (tensor.Float64Tensor, error),
) (tensor.Float64Tensor, error) {
	if len(inputs) != count {
		return nil, fmt.Errorf("cpu tensor: %s node %q requires %d inputs", node.Op, node.ID, count)
	}

	var second tensor.Float64Tensor
	if len(inputs) > 1 {
		second = inputs[1]
	}

	return apply(inputs[0], second)
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

func intSliceConfig(node executor.NodeSpec, key string) []int {
	value, ok := node.Metadata[key]

	if !ok {
		return nil
	}

	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...)
	case []int64:
		values := make([]int, len(typed))
		for index, item := range typed {
			values[index] = int(item)
		}
		return values
	case []float64:
		values := make([]int, len(typed))
		for index, item := range typed {
			values[index] = int(item)
		}
		return values
	default:
		return nil
	}
}

func floatSliceConfig(node executor.NodeSpec, key string) []float64 {
	value, ok := node.Metadata[key]

	if !ok {
		return nil
	}

	switch typed := value.(type) {
	case []float64:
		return append([]float64(nil), typed...)
	case []float32:
		values := make([]float64, len(typed))
		for index, item := range typed {
			values[index] = float64(item)
		}
		return values
	default:
		return nil
	}
}
