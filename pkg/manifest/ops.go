package manifest

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/convolution"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/embedding"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/masking"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/pooling"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/positional"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/projection"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/shape"
)

func init() {
	registerActivation()
	registerAttention()
	registerConvolution()
	registerEmbedding()
	registerMasking()
	registerMath()
	registerPooling()
	registerPositional()
	registerProjection()
	registerShape()
}

func registerActivation() {
	Register("activation.relu", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewReLU(), nil
	})
	Register("activation.gelu", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewGelu(), nil
	})
	Register("activation.sigmoid", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewSigmoid(), nil
	})
	Register("activation.tanh", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewTanh(), nil
	})
	Register("activation.swiglu", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewSwiGLU(), nil
	})
	Register("activation.leaky_relu", func(config map[string]any) (operation.Operation, error) {
		alpha, _ := config["alpha"].(float64)
		return activation.NewLeakyReLU(alpha), nil
	})
}

func registerAttention() {
	Register("attention.sdpa", func(_ map[string]any) (operation.Operation, error) {
		return attention.NewSDPA(), nil
	})
	Register("attention.gqa", func(_ map[string]any) (operation.Operation, error) {
		return attention.NewGQA(), nil
	})
	Register("attention.mqa", func(_ map[string]any) (operation.Operation, error) {
		return attention.NewMQA(), nil
	})
	Register("attention.sliding_window", func(config map[string]any) (operation.Operation, error) {
		window, _ := config["window"].(int)
		return attention.NewSlidingWindow(window), nil
	})
}

func registerConvolution() {
	Register("convolution.conv1d", func(config map[string]any) (operation.Operation, error) {
		return convolution.NewConv1d(
			intParam(config, "in_channels"),
			intParam(config, "out_channels"),
			intParam(config, "kernel_size"),
			intParamDefault(config, "stride", 1),
			intParam(config, "padding"),
			intParamDefault(config, "dilation", 1),
			intParamDefault(config, "groups", 1),
		), nil
	})
	Register("convolution.conv2d", func(config map[string]any) (operation.Operation, error) {
		return convolution.NewConv2d(
			intParam(config, "in_channels"),
			intParam(config, "out_channels"),
			intParam(config, "kernel_h"),
			intParam(config, "kernel_w"),
			intParamDefault(config, "stride_h", 1),
			intParamDefault(config, "stride_w", 1),
			intParam(config, "pad_h"),
			intParam(config, "pad_w"),
			intParamDefault(config, "dil_h", 1),
			intParamDefault(config, "dil_w", 1),
			intParamDefault(config, "groups", 1),
		), nil
	})
	Register("convolution.conv3d", func(config map[string]any) (operation.Operation, error) {
		return convolution.NewConv3d(
			intParam(config, "in_channels"),
			intParam(config, "out_channels"),
			intParam(config, "kernel_d"),
			intParam(config, "kernel_h"),
			intParam(config, "kernel_w"),
			intParamDefault(config, "stride_d", 1),
			intParamDefault(config, "stride_h", 1),
			intParamDefault(config, "stride_w", 1),
			intParam(config, "pad_d"),
			intParam(config, "pad_h"),
			intParam(config, "pad_w"),
			intParamDefault(config, "dil_d", 1),
			intParamDefault(config, "dil_h", 1),
			intParamDefault(config, "dil_w", 1),
			intParamDefault(config, "groups", 1),
		), nil
	})
	Register("convolution.conv_transpose2d", func(config map[string]any) (operation.Operation, error) {
		return convolution.NewConvTranspose2d(
			intParam(config, "in_channels"),
			intParam(config, "out_channels"),
			intParam(config, "kernel_h"),
			intParam(config, "kernel_w"),
			intParamDefault(config, "stride_h", 1),
			intParamDefault(config, "stride_w", 1),
			intParam(config, "pad_h"),
			intParam(config, "pad_w"),
			intParam(config, "out_pad_h"),
			intParam(config, "out_pad_w"),
			intParamDefault(config, "dil_h", 1),
			intParamDefault(config, "dil_w", 1),
			intParamDefault(config, "groups", 1),
		), nil
	})
}

func registerEmbedding() {
	Register("embedding.token", func(config map[string]any) (operation.Operation, error) {
		vocabSize := intParam(config, "vocab_size")
		dModel := intParam(config, "d_model")
		initStd, _ := config["init_std"].(float64)

		if initStd == 0 {
			initStd = 0.02
		}

		return embedding.NewTokenEmbedding(vocabSize, dModel, initStd), nil
	})
}

func registerMasking() {
	Register("masking.causal", func(_ map[string]any) (operation.Operation, error) {
		return masking.NewCausalMask(), nil
	})
	Register("masking.apply", func(_ map[string]any) (operation.Operation, error) {
		return masking.NewApplyMask(), nil
	})
}

func registerMath() {
	Register("math.add", func(_ map[string]any) (operation.Operation, error) {
		return math.NewAdd(), nil
	})
	Register("math.mul", func(_ map[string]any) (operation.Operation, error) {
		return math.NewMul(), nil
	})
	Register("math.matmul", func(_ map[string]any) (operation.Operation, error) {
		return math.NewMatmul(), nil
	})
	Register("math.softmax", func(_ map[string]any) (operation.Operation, error) {
		return math.NewSoftmax(), nil
	})
	Register("math.layernorm", func(config map[string]any) (operation.Operation, error) {
		eps, _ := config["eps"].(float64)
		if eps == 0 {
			eps = 1e-5
		}
		return math.NewLayerNorm(eps, nil, nil), nil
	})
	Register("math.rmsnorm", func(config map[string]any) (operation.Operation, error) {
		eps, _ := config["eps"].(float64)
		if eps == 0 {
			eps = 1e-6
		}
		return math.NewRMSNorm(eps, nil), nil
	})
	Register("math.dropout", func(config map[string]any) (operation.Operation, error) {
		p, _ := config["p"].(float64)
		training, _ := config["training"].(bool)
		return math.NewDropout(p, training), nil
	})
	Register("math.exp", func(_ map[string]any) (operation.Operation, error) {
		return math.NewExp(), nil
	})
	Register("math.log", func(_ map[string]any) (operation.Operation, error) {
		return math.NewLog(), nil
	})
	Register("math.logsumexp", func(_ map[string]any) (operation.Operation, error) {
		return math.NewLogSumExp(), nil
	})
	Register("math.inv_sqrt_dim_scale", func(_ map[string]any) (operation.Operation, error) {
		return math.NewInvSqrtDimScale(), nil
	})
	Register("math.sign", func(_ map[string]any) (operation.Operation, error) {
		return math.NewSign(), nil
	})
	Register("math.outer", func(_ map[string]any) (operation.Operation, error) {
		return math.NewOuter(), nil
	})
}

func registerPooling() {
	Register("pooling.avg_pool2d", func(config map[string]any) (operation.Operation, error) {
		return pooling.NewAvgPool2d(
			intParam(config, "kernel_h"),
			intParam(config, "kernel_w"),
			intParamDefault(config, "stride_h", 1),
			intParamDefault(config, "stride_w", 1),
			intParam(config, "pad_h"),
			intParam(config, "pad_w"),
			intParamDefault(config, "dil_h", 1),
			intParamDefault(config, "dil_w", 1),
			boolParam(config, "ceil"),
			boolParamDefault(config, "count_include_pad", true),
			intParam(config, "divisor_override"),
		), nil
	})
	Register("pooling.max_pool2d", func(config map[string]any) (operation.Operation, error) {
		return pooling.NewMaxPool2d(
			intParam(config, "kernel_h"),
			intParam(config, "kernel_w"),
			intParamDefault(config, "stride_h", 1),
			intParamDefault(config, "stride_w", 1),
			intParam(config, "pad_h"),
			intParam(config, "pad_w"),
			intParamDefault(config, "dil_h", 1),
			intParamDefault(config, "dil_w", 1),
			boolParam(config, "ceil"),
		), nil
	})
	Register("pooling.adaptive_avg_pool2d", func(config map[string]any) (operation.Operation, error) {
		return pooling.NewAdaptiveAvgPool2d(
			intParam(config, "out_h"),
			intParam(config, "out_w"),
		), nil
	})
	Register("pooling.adaptive_max_pool2d", func(config map[string]any) (operation.Operation, error) {
		return pooling.NewAdaptiveMaxPool2d(
			intParam(config, "out_h"),
			intParam(config, "out_w"),
		), nil
	})
}

func registerPositional() {
	Register("positional.rope", func(config map[string]any) (operation.Operation, error) {
		base, _ := config["base"].(float64)
		if base == 0 {
			base = 10000
		}
		return positional.NewRoPE(base, intParam(config, "head_dim")), nil
	})
	Register("positional.alibi", func(config map[string]any) (operation.Operation, error) {
		causal, _ := config["causal"].(bool)
		return positional.NewALiBi(intParam(config, "num_heads"), causal), nil
	})
}

func registerProjection() {
	Register("projection.linear", func(config map[string]any) (operation.Operation, error) {
		return projection.NewLinear(
			intParam(config, "in_features"),
			intParam(config, "out_features"),
		), nil
	})
	Register("projection.fused_qkv", func(config map[string]any) (operation.Operation, error) {
		return projection.NewFusedQKV(
			intParam(config, "d_in"),
			intParam(config, "d_q"),
			intParam(config, "d_k"),
			intParam(config, "d_v"),
		), nil
	})
}

func registerShape() {
	Register("shape.transpose", func(config map[string]any) (operation.Operation, error) {
		return shape.NewTranspose(
			intParam(config, "dim0"),
			intParam(config, "dim1"),
		), nil
	})
	Register("shape.reshape", func(config map[string]any) (operation.Operation, error) {
		raw, _ := config["shape"].([]any)
		target := make([]int, len(raw))

		for idx, val := range raw {
			target[idx] = anyToInt(val)
		}

		return shape.NewReshape(target), nil
	})
	Register("shape.concat", func(config map[string]any) (operation.Operation, error) {
		return shape.NewConcat(intParam(config, "dim")), nil
	})
	Register("shape.split", func(config map[string]any) (operation.Operation, error) {
		return shape.NewSplit(
			intParam(config, "split_size"),
			intParam(config, "dim"),
		), nil
	})
	Register("shape.view_as_heads", func(config map[string]any) (operation.Operation, error) {
		return shape.NewViewAsHeads(intParam(config, "num_heads")), nil
	})
	Register("shape.merge_heads", func(_ map[string]any) (operation.Operation, error) {
		return shape.NewMergeHeads(), nil
	})
}

func intParam(config map[string]any, key string) int {
	return anyToInt(config[key])
}

func intParamDefault(config map[string]any, key string, defaultVal int) int {
	val, ok := config[key]

	if !ok {
		return defaultVal
	}

	result := anyToInt(val)

	if result == 0 {
		return defaultVal
	}

	return result
}

func boolParam(config map[string]any, key string) bool {
	val, _ := config[key].(bool)
	return val
}

func boolParamDefault(config map[string]any, key string, defaultVal bool) bool {
	val, ok := config[key]

	if !ok {
		return defaultVal
	}

	result, _ := val.(bool)
	return result
}

func anyToInt(val any) int {
	switch typed := val.(type) {
	case int:
		return typed
	case float64:
		return int(typed)
	case int64:
		return int(typed)
	default:
		return 0
	}
}

