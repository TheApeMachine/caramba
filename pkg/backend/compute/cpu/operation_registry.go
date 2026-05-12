package cpu

import (
	"context"
	"fmt"

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

/*
NewOperationRegistry publishes the CPU operation surface to the shared executor.
*/
func NewOperationRegistry() *executor.Registry {
	registry := executor.NewTensorRegistry()

	registerCPU(registry, "activation.swish", math.NewMul())
	registerCPU(registry, "attention.sdpa", attention.NewSDPA())
	registerCPU(registry, "attention.mqa", attention.NewMQA())
	registerCPU(registry, "attention.gqa", attention.NewGQA())
	registerFactory(registry, "attention.sliding_window", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(attention.NewSlidingWindow(intConfig(node, "window", 1)))
	})
	registerCPU(registry, "masking.apply", masking.NewApplyMask())
	registerCPU(registry, "masking.causal", masking.NewCausalMask())
	registerCPU(registry, "math.exp", math.NewExp())
	registerCPU(registry, "math.log", math.NewLog())
	registerCPU(registry, "math.logsumexp", math.NewLogSumExp())
	registerCPU(registry, "math.softmax", math.NewSoftmax())
	registerCPU(registry, "math.outer", math.NewOuter())
	registerCPU(registry, "math.sign", math.NewSign())
	registerCPU(registry, "math.inv_sqrt_dim_scale", math.NewInvSqrtDimScale())
	registerFactory(registry, "math.dropout", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(math.NewDropout(
			floatConfig(node, "p", 0),
			boolConfig(node, "training", false),
		))
	})
	registerFactory(registry, "math.rmsnorm", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(math.NewRMSNorm(
			floatConfig(node, "eps", 1e-5),
			floatSliceConfig(node, "weight"),
		))
	})
	registerFactory(registry, "math.layernorm", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(math.NewLayerNorm(
			floatConfig(node, "eps", 1e-5),
			floatSliceConfig(node, "weight"),
			floatSliceConfig(node, "bias"),
		))
	})
	registerFactory(registry, "shape.reshape", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(shape.NewReshape(intSliceConfig(node, "shape")))
	})
	registerFactory(registry, "shape.transpose", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(shape.NewTranspose(
			intConfig(node, "dim0", 0),
			intConfig(node, "dim1", 1),
		))
	})
	registerFactory(registry, "shape.concat", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(shape.NewConcat(intConfig(node, "dim", 0)))
	})
	registerFactory(registry, "shape.split", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(shape.NewSplit(
			intConfig(node, "split_size", 1),
			intConfig(node, "dim", 0),
		))
	})
	registerFactory(registry, "shape.view_as_heads", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(shape.NewViewAsHeads(intConfig(node, "num_heads", 1)))
	})
	registerCPU(registry, "shape.merge_heads", shape.NewMergeHeads())
	registerFactory(registry, "positional.rope", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(positional.NewRoPE(
			floatConfig(node, "base", 10000),
			intConfig(node, "head_dim", 1),
		))
	})
	registerFactory(registry, "positional.alibi", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(positional.NewALiBi(
			intConfig(node, "num_heads", 1),
			boolConfig(node, "causal", true),
		))
	})
	registerFactory(registry, "convolution.conv1d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(convolution.NewConv1d(
			intConfig(node, "in_c", 1),
			intConfig(node, "out_c", 1),
			intConfig(node, "kernel_size", 1),
			intConfig(node, "stride", 1),
			intConfig(node, "padding", 0),
			intConfig(node, "dilation", 1),
			intConfig(node, "groups", 1),
		))
	})
	registerFactory(registry, "convolution.conv2d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(convolution.NewConv2d(
			intConfig(node, "in_c", 1),
			intConfig(node, "out_c", 1),
			intConfig(node, "k_h", 1),
			intConfig(node, "k_w", 1),
			intConfig(node, "stride_h", 1),
			intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0),
			intConfig(node, "pad_w", 0),
			intConfig(node, "dil_h", 1),
			intConfig(node, "dil_w", 1),
			intConfig(node, "groups", 1),
		))
	})
	registerFactory(registry, "convolution.conv3d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(convolution.NewConv3d(
			intConfig(node, "in_c", 1), intConfig(node, "out_c", 1),
			intConfig(node, "k_d", 1), intConfig(node, "k_h", 1), intConfig(node, "k_w", 1),
			intConfig(node, "s_d", 1), intConfig(node, "s_h", 1), intConfig(node, "s_w", 1),
			intConfig(node, "p_d", 0), intConfig(node, "p_h", 0), intConfig(node, "p_w", 0),
			intConfig(node, "dil_d", 1), intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			intConfig(node, "groups", 1),
		))
	})
	registerFactory(registry, "convolution.conv_transpose2d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(convolution.NewConvTranspose2d(
			intConfig(node, "in_c", 1), intConfig(node, "out_c", 1),
			intConfig(node, "k_h", 1), intConfig(node, "k_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "out_pad_h", 0), intConfig(node, "out_pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			intConfig(node, "groups", 1),
		))
	})
	registerFactory(registry, "pooling.max_pool2d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(pooling.NewMaxPool2d(
			intConfig(node, "kernel_h", 1), intConfig(node, "kernel_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			boolConfig(node, "ceil", false),
		))
	})
	registerFactory(registry, "pooling.avg_pool2d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(pooling.NewAvgPool2d(
			intConfig(node, "kernel_h", 1), intConfig(node, "kernel_w", 1),
			intConfig(node, "stride_h", 1), intConfig(node, "stride_w", 1),
			intConfig(node, "pad_h", 0), intConfig(node, "pad_w", 0),
			intConfig(node, "dil_h", 1), intConfig(node, "dil_w", 1),
			boolConfig(node, "ceil", false), boolConfig(node, "count_include_pad", false),
			intConfig(node, "divisor_override", 0),
		))
	})
	registerFactory(registry, "pooling.adaptive_avg_pool2d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(pooling.NewAdaptiveAvgPool2d(
			intConfig(node, "out_h", 1),
			intConfig(node, "out_w", 1),
		))
	})
	registerFactory(registry, "pooling.adaptive_max_pool2d", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(pooling.NewAdaptiveMaxPool2d(
			intConfig(node, "out_h", 1),
			intConfig(node, "out_w", 1),
		))
	})
	registerFactory(registry, "projection.linear", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(projection.NewLinear(
			intConfig(node, "in_features", 1),
			intConfig(node, "out_features", 1),
		))
	})
	registerFactory(registry, "projection.fused_qkv", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(projection.NewFusedQKV(
			intConfig(node, "d_in", 1), intConfig(node, "d_q", 1),
			intConfig(node, "d_k", 1), intConfig(node, "d_v", 1),
		))
	})
	registerCPU(registry, "hawkes.intensity", hawkes.NewIntensity())
	registerCPU(registry, "hawkes.kernel_matrix", hawkes.NewKernelMatrix())
	registerCPU(registry, "hawkes.log_likelihood", hawkes.NewLogLikelihood())
	registerCPU(registry, "hawkes.simulate", hawkes.NewSimulate())
	registerCPU(registry, "vsa.bind", vsa.NewBind())
	registerCPU(registry, "vsa.bundle", vsa.NewBundle())
	registerCPU(registry, "vsa.similarity", vsa.NewSimilarity())
	registerFactory(registry, "vsa.permute", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(vsa.NewPermute(intConfig(node, "k", 1)))
	})
	registerFactory(registry, "vsa.inverse_permute", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(vsa.NewInversePermute(intConfig(node, "k", 1)))
	})
	registerCPU(registry, "active_inference.belief_update", active_inference.NewBeliefUpdate())
	registerCPU(registry, "active_inference.expected_free_energy", active_inference.NewExpectedFreeEnergy())
	registerCPU(registry, "active_inference.free_energy", active_inference.NewFreeEnergy())
	registerCPU(registry, "active_inference.precision_weight", active_inference.NewPrecisionWeight())
	registerCPU(registry, "predictive_coding.prediction", predictive_coding.NewPrediction())
	registerCPU(registry, "predictive_coding.prediction_error", predictive_coding.NewPredictionError())
	registerCPU(registry, "predictive_coding.update_representation", predictive_coding.NewUpdateRepresentation())
	registerCPU(registry, "predictive_coding.update_weights", predictive_coding.NewUpdateWeights())
	registerCPU(registry, "markov_blanket.flow_active", markov_blanket.NewFlowActive())
	registerCPU(registry, "markov_blanket.flow_internal", markov_blanket.NewFlowInternal())
	registerCPU(registry, "markov_blanket.mutual_information", markov_blanket.NewMutualInformation())
	registerCPU(registry, "markov_blanket.partition", markov_blanket.NewPartition())
	registerCPU(registry, "causal.backdoor_adjustment", causal.NewBackdoorAdjustment())
	registerCPU(registry, "causal.cate", causal.NewCATE())
	registerCPU(registry, "causal.counterfactual", causal.NewCounterfactual())
	registerCPU(registry, "causal.dag_markov_factorization", causal.NewDAGMarkovFactorization())
	registerCPU(registry, "causal.do_calculus", causal.NewDoCalculus())
	registerCPU(registry, "causal.frontdoor_adjustment", causal.NewFrontdoorAdjustment())
	registerCPU(registry, "causal.iv_estimate", causal.NewIVEstimate())
	registerCPU(registry, "train.loss.mse", train.NewMSELoss())
	registerCPU(registry, "train.loss.cross_entropy", train.NewCrossEntropyLoss())
	registerCPU(registry, "train.loss.mse_grad", train.NewMSEGrad())
	registerCPU(registry, "train.loss.cross_entropy_grad", train.NewCrossEntropyGrad())
	registerFactory(registry, "train.optimizer.adam", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(train.NewAdamStep(
			floatConfig(node, "lr", 1e-3), floatConfig(node, "beta1", 0.9),
			floatConfig(node, "beta2", 0.999), floatConfig(node, "eps", 1e-8),
			floatConfig(node, "wd", 0),
		))
	})
	registerFactory(registry, "train.optimizer.adamw", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(train.NewAdamWStep(
			floatConfig(node, "lr", 1e-3), floatConfig(node, "beta1", 0.9),
			floatConfig(node, "beta2", 0.999), floatConfig(node, "eps", 1e-8),
			floatConfig(node, "wd", 0),
		))
	})
	registerFactory(registry, "train.optimizer.sgd", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(train.NewSGDStep(
			floatConfig(node, "lr", 1e-3), floatConfig(node, "momentum", 0),
			floatConfig(node, "wd", 0), boolConfig(node, "nesterov", false),
		))
	})
	registerCPU(registry, "bench.accuracy", bench.NewAccuracy())
	registerCPU(registry, "bench.perplexity", bench.NewPerplexity())
	registerCPU(registry, "bench.f1", bench.NewF1())
	registerFactory(registry, "model.graft", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(model.NewGraft(
			stringConfig(node, "source", ""),
			stringConfig(node, "at", ""),
			stringConfig(node, "mode", "append"),
		))
	})
	registerFactory(registry, "model.freeze", func(node executor.NodeSpec) executor.Handler {
		return executor.OperationHandler(model.NewFreeze(
			stringConfig(node, "source", ""),
			stringConfig(node, "pattern", ""),
			stringConfig(node, "except", ""),
			boolConfig(node, "frozen", true),
		))
	})

	return registry
}

func registerCPU(registry *executor.Registry, op string, operation interface {
	Forward([]int, ...[]float64) []float64
}) {
	registry.Register(ir.OpType(op), executor.OperationHandler(operation))
}

func registerFactory(
	registry *executor.Registry,
	op string,
	factory func(executor.NodeSpec) executor.Handler,
) {
	registry.Register(ir.OpType(op), func(
		ctx context.Context,
		backend executor.Backend,
		node executor.NodeSpec,
		inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		handler := factory(node)

		if handler == nil {
			return nil, fmt.Errorf("cpu registry: operation %s returned no handler", op)
		}

		return handler(ctx, backend, node, inputs)
	})
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
