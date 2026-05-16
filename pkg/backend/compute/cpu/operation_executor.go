package cpu

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/bench"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/convolution"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/embedding"
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
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type OperationDispatchContract struct{}

var TensorOperationDispatchContract = OperationDispatchContract{}

func (operationDispatchContract OperationDispatchContract) SupportedIDSet() map[ir.OpType]bool {
	return map[ir.OpType]bool{
		ir.OpInput:                              true,
		ir.OpAdd:                                true,
		ir.OpMul:                                true,
		ir.OpMatmul:                             true,
		ir.OpReLU:                               true,
		ir.OpLeakyReLU:                          true,
		ir.OpGELU:                               true,
		ir.OpTanh:                               true,
		ir.OpSigmoid:                            true,
		ir.OpSwiGLU:                             true,
		ir.OpSwish:                              true,
		ir.OpSELU:                               true,
		ir.OpFused:                              true,
		"activation.relu":                       true,
		"activation.leaky_relu":                 true,
		"activation.gelu":                       true,
		"activation.tanh":                       true,
		"activation.sigmoid":                    true,
		"activation.swiglu":                     true,
		"activation.swish":                      true,
		"activation.selu":                       true,
		"attention.sdpa":                        true,
		"attention.mqa":                         true,
		"attention.gqa":                         true,
		"attention.sliding_window":              true,
		"masking.apply":                         true,
		"masking.causal":                        true,
		"math.add":                              true,
		"math.mul":                              true,
		"math.matmul":                           true,
		"math.exp":                              true,
		"math.log":                              true,
		"math.logsumexp":                        true,
		"math.softmax":                          true,
		"math.outer":                            true,
		"math.sign":                             true,
		"math.inv_sqrt_dim_scale":               true,
		"math.dropout":                          true,
		"math.rmsnorm":                          true,
		"math.layernorm":                        true,
		"shape.reshape":                         true,
		"shape.transpose":                       true,
		"shape.concat":                          true,
		"shape.split":                           true,
		"shape.view_as_heads":                   true,
		"shape.last_token":                      true,
		"shape.merge_heads":                     true,
		"positional.rope":                       true,
		"positional.alibi":                      true,
		"embedding.token":                       true,
		"convolution.conv1d":                    true,
		"convolution.conv2d":                    true,
		"convolution.conv3d":                    true,
		"convolution.conv_transpose2d":          true,
		"pooling.max_pool2d":                    true,
		"pooling.avg_pool2d":                    true,
		"pooling.adaptive_avg_pool2d":           true,
		"pooling.adaptive_max_pool2d":           true,
		"projection.linear":                     true,
		"projection.fused_qkv":                  true,
		"hawkes.intensity":                      true,
		"hawkes.kernel_matrix":                  true,
		"hawkes.log_likelihood":                 true,
		"hawkes.simulate":                       true,
		"vsa.bind":                              true,
		"vsa.bundle":                            true,
		"vsa.similarity":                        true,
		"vsa.permute":                           true,
		"vsa.inverse_permute":                   true,
		"active_inference.belief_update":        true,
		"active_inference.expected_free_energy": true,
		"active_inference.free_energy":          true,
		"active_inference.precision_weight":     true,
		"predictive_coding.prediction":          true,
		"predictive_coding.prediction_error":    true,
		"predictive_coding.update_representation": true,
		"predictive_coding.update_weights":        true,
		"markov_blanket.flow_active":              true,
		"markov_blanket.flow_internal":            true,
		"markov_blanket.mutual_information":       true,
		"markov_blanket.partition":                true,
		"causal.backdoor_adjustment":              true,
		"causal.cate":                             true,
		"causal.counterfactual":                   true,
		"causal.dag_markov_factorization":         true,
		"causal.do_calculus":                      true,
		"causal.frontdoor_adjustment":             true,
		"causal.iv_estimate":                      true,
		"train.loss.mse":                          true,
		"train.loss.cross_entropy":                true,
		"train.loss.mse_grad":                     true,
		"train.loss.cross_entropy_grad":           true,
		"train.grad.mse":                          true,
		"train.grad.cross_entropy":                true,
		"train.optimizer.adam":                    true,
		"train.optimizer.adamw":                   true,
		"train.optimizer.sgd":                     true,
		"train.optimizer.lion":                    true,
		"train.optimizer.rmsprop":                 true,
		"bench.accuracy":                          true,
		"bench.perplexity":                        true,
		"bench.f1":                                true,
		"bench.metric.accuracy":                   true,
		"bench.metric.perplexity":                 true,
		"bench.metric.f1":                         true,
		"model.graft":                             true,
		"model.freeze":                            true,
	}
}

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
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewAdd())
	case ir.OpMul:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewMul())
	case ir.OpMatmul:
		return tensorBackend.applyMatmul(ctx, node, inputs)
	case ir.OpReLU:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewReLU())
	case ir.OpLeakyReLU:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewLeakyReLU())
	case ir.OpGELU:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewGelu())
	case ir.OpTanh:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewTanh())
	case ir.OpSigmoid:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewSigmoid())
	case ir.OpSwish:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewSwish())
	case ir.OpSwiGLU:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewSwiGLU())
	case ir.OpSELU:
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewSELU())
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

	activationName, _ := node.Metadata["activation"].(string)
	if len(inputs) == 2 {
		output, err := tensorBackend.applyMatmul(ctx, node, inputs)

		if err != nil {
			return nil, err
		}

		switch {
		case strings.EqualFold(activationName, string(ir.OpReLU)):
			return tensorBackend.applyActivation(ctx, node, output, activation.NewReLU())
		case strings.EqualFold(activationName, string(ir.OpGELU)):
			return tensorBackend.applyActivation(ctx, node, output, activation.NewGelu())
		default:
			return output, nil
		}
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("cpu tensor: Fused node %q requires 2 or 3 inputs", node.ID)
	}

	output, err := tensorBackend.applyMatmulAdd(ctx, node, inputs)

	if err != nil {
		return nil, err
	}

	if strings.EqualFold(activationName, string(ir.OpGELU)) {
		return tensorBackend.applyActivation(ctx, node, output, activation.NewGelu())
	}

	return output, nil
}

func (tensorBackend *TensorBackend) applyOperation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	switch strings.ToLower(string(node.Op)) {
	case "activation.swish":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewSwish())
	case "activation.selu":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, activation.NewSELU())
	case "attention.sdpa":
		return tensorBackend.applySDPA(ctx, node, inputs)
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
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewDropout())
	case "math.rmsnorm":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewRMSNorm())
	case "math.layernorm":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, math.NewLayerNorm())
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
	case "shape.last_token":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, shape.NewLastToken())
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
	case "embedding.token":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, embedding.NewTokenEmbedding(
			intConfig(node, "vocab_size", 1),
			intConfig(node, "d_model", 1),
			floatConfig(node, "init_std", 0.02),
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
	case "train.loss.mse_grad", "train.grad.mse":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewMSEGrad())
	case "train.loss.cross_entropy_grad", "train.grad.cross_entropy":
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
	case "train.optimizer.lion":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewLionStep(
			floatConfig(node, "lr", 1e-4), floatConfig(node, "beta1", 0.9),
			floatConfig(node, "beta2", 0.99), floatConfig(node, "wd", 0),
		))
	case "train.optimizer.rmsprop":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, train.NewRMSPropStep(
			floatConfig(node, "lr", 1e-2), floatConfig(node, "alpha", 0.99),
			floatConfig(node, "eps", 1e-8), floatConfig(node, "momentum", 0),
			floatConfig(node, "wd", 0),
		))
	case "bench.accuracy", "bench.metric.accuracy":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, bench.NewAccuracy())
	case "bench.perplexity", "bench.metric.perplexity":
		return executor.RunOperation(ctx, tensorBackend, node, inputs, bench.NewPerplexity())
	case "bench.f1", "bench.metric.f1":
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

func (tensorBackend *TensorBackend) applyMatmul(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("cpu tensor: %s node %q requires 2 inputs", node.Op, node.ID)
	}

	operationNode, err := withMatmulOperationShape(node, inputs)

	if err != nil {
		return nil, err
	}

	return executor.RunOperation(ctx, tensorBackend, operationNode, inputs, math.NewMatmul())
}

func (tensorBackend *TensorBackend) applyMatmulAdd(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("cpu tensor: %s node %q requires 3 inputs", node.Op, node.ID)
	}

	operationNode, err := withMatmulOperationShape(node, inputs)

	if err != nil {
		return nil, err
	}

	return executor.RunOperation(ctx, tensorBackend, operationNode, inputs, math.NewMatmulAdd())
}

func (tensorBackend *TensorBackend) applyActivation(
	ctx context.Context,
	node executor.NodeSpec,
	input tensor.Float64Tensor,
	operation interface {
		Forward(*state.Dict) (*state.Dict, error)
	},
) (tensor.Float64Tensor, error) {
	output, err := executor.RunOperation(ctx, tensorBackend, node, []tensor.Float64Tensor{input}, operation)

	if err != nil {
		_ = input.Close()

		return nil, err
	}

	if closeErr := input.Close(); closeErr != nil {
		return nil, closeErr
	}

	return output, nil
}

func withMatmulOperationShape(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (executor.NodeSpec, error) {
	if len(inputs) < 2 {
		return executor.NodeSpec{}, fmt.Errorf("cpu tensor: %s node %q requires at least 2 inputs", node.Op, node.ID)
	}

	leftDims := inputs[0].Shape().Dims()
	rightDims := inputs[1].Shape().Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return executor.NodeSpec{}, fmt.Errorf("cpu tensor: %s node %q requires rank-2 matrices", node.Op, node.ID)
	}

	if leftDims[1] != rightDims[0] {
		return executor.NodeSpec{}, fmt.Errorf(
			"cpu tensor: %s node %q dimension mismatch [%d,%d] x [%d,%d]",
			node.Op, node.ID, leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	metadata := make(map[string]any, len(node.Metadata)+1)

	for key, value := range node.Metadata {
		metadata[key] = value
	}

	metadata["op_shape"] = []int{leftDims[0], leftDims[1], rightDims[1]}
	node.Metadata = metadata

	return node, nil
}

func (tensorBackend *TensorBackend) applySDPA(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	cache, _ := node.Metadata["kv_cache"].(*kv.Cache)

	if cache == nil {
		return executor.RunOperation(ctx, tensorBackend, node, inputs, attention.NewSDPA())
	}

	if !boolConfig(node, "causal", false) {
		return nil, fmt.Errorf("cpu tensor: KV cache requires causal SDPA node %q", node.ID)
	}

	return tensorBackend.applyCachedSDPA(ctx, node, inputs, cache)
}

func (tensorBackend *TensorBackend) applyCachedSDPA(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
	cache *kv.Cache,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("cpu tensor: SDPA node %q requires 3 inputs", node.ID)
	}

	values, err := executor.InputValues(tensorBackend, inputs)

	if err != nil {
		return nil, err
	}

	keyShape := inputs[1].Shape().Dims()
	cachedKey, cachedValue, _, err := cache.Append(node.ID, keyShape, values[1], values[2])

	if err != nil {
		return nil, err
	}

	outputState, err := attention.NewSDPA().Forward(
		executor.OperationState(
			node,
			inputs,
			[][]float64{values[0], cachedKey, cachedValue},
		),
	)

	if err != nil {
		return nil, err
	}

	return executor.UploadOutput(tensorBackend, node, inputs, outputState.Out)
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
