package manifest

import (
	"fmt"
	"strconv"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/bench"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/convolution"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/data"
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
	tokenizerop "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/tokenizer"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/train"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/vsa"
)

func init() {
	registerActivation()
	registerActiveInference()
	registerAttention()
	registerBench()
	registerCausal()
	registerConvolution()
	registerData()
	registerEmbedding()
	registerHawkes()
	registerMarkovBlanket()
	registerMasking()
	registerMath()
	registerModel()
	registerPooling()
	registerPositional()
	registerPredictiveCoding()
	registerProjection()
	registerShape()
	registerTokenizer()
	registerTrain()
	registerVSA()
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
	Register("activation.swish", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewSwish(), nil
	})
	Register("activation.selu", func(_ map[string]any) (operation.Operation, error) {
		return activation.NewSELU(), nil
	})
	Register("activation.leaky_relu", func(config map[string]any) (operation.Operation, error) {
		return activation.NewLeakyReLU(), nil
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
		window, err := requiredIntParam(config, "window")

		if err != nil {
			return nil, fmt.Errorf("attention.sliding_window config.window: %w", err)
		}

		return attention.NewSlidingWindow(window), nil
	})
}

func registerConvolution() {
	Register("convolution.conv1d", func(config map[string]any) (operation.Operation, error) {
		inChannels, err := requiredIntParam(config, "in_channels")

		if err != nil {
			return nil, fmt.Errorf("convolution.conv1d config.in_channels: %w", err)
		}

		outChannels, err := requiredIntParam(config, "out_channels")

		if err != nil {
			return nil, fmt.Errorf("convolution.conv1d config.out_channels: %w", err)
		}

		kernelSize, err := requiredIntParam(config, "kernel_size")

		if err != nil {
			return nil, fmt.Errorf("convolution.conv1d config.kernel_size: %w", err)
		}

		padding, err := requiredIntParam(config, "padding")

		if err != nil {
			return nil, fmt.Errorf("convolution.conv1d config.padding: %w", err)
		}

		return convolution.NewConv1d(
			inChannels,
			outChannels,
			kernelSize,
			intParamDefault(config, "stride", 1),
			padding,
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
		vocabSize, err := requiredIntParam(config, "vocab_size")

		if err != nil {
			return nil, fmt.Errorf("embedding.token config.vocab_size: %w", err)
		}

		dModel, err := requiredIntParam(config, "d_model")

		if err != nil {
			return nil, fmt.Errorf("embedding.token config.d_model: %w", err)
		}

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
	Register("math.matmul_add", func(_ map[string]any) (operation.Operation, error) {
		return math.NewMatmulAdd(), nil
	})
	Register("math.softmax", func(_ map[string]any) (operation.Operation, error) {
		return math.NewSoftmax(), nil
	})
	Register("math.layernorm", func(config map[string]any) (operation.Operation, error) {
		return math.NewLayerNorm(), nil
	})
	Register("math.rmsnorm", func(config map[string]any) (operation.Operation, error) {
		return math.NewRMSNorm(), nil
	})
	Register("math.groupnorm", func(config map[string]any) (operation.Operation, error) {
		return math.NewGroupNorm(), nil
	})
	Register("math.dropout", func(config map[string]any) (operation.Operation, error) {
		return math.NewDropout(), nil
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
		outH, err := requiredIntParam(config, "out_h")

		if err != nil {
			return nil, fmt.Errorf("pooling.adaptive_avg_pool2d config.out_h: %w", err)
		}

		outW, err := requiredIntParam(config, "out_w")

		if err != nil {
			return nil, fmt.Errorf("pooling.adaptive_avg_pool2d config.out_w: %w", err)
		}

		return pooling.NewAdaptiveAvgPool2d(
			outH,
			outW,
		), nil
	})
	Register("pooling.adaptive_max_pool2d", func(config map[string]any) (operation.Operation, error) {
		outH, err := requiredIntParam(config, "out_h")

		if err != nil {
			return nil, fmt.Errorf("pooling.adaptive_max_pool2d config.out_h: %w", err)
		}

		outW, err := requiredIntParam(config, "out_w")

		if err != nil {
			return nil, fmt.Errorf("pooling.adaptive_max_pool2d config.out_w: %w", err)
		}

		return pooling.NewAdaptiveMaxPool2d(
			outH,
			outW,
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
		inFeatures, err := requiredIntParam(config, "in_features")

		if err != nil {
			return nil, fmt.Errorf("projection.linear config.in_features: %w", err)
		}

		outFeatures, err := requiredIntParam(config, "out_features")

		if err != nil {
			return nil, fmt.Errorf("projection.linear config.out_features: %w", err)
		}

		return projection.NewLinear(
			inFeatures,
			outFeatures,
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
		raw, ok := config["shape"].([]any)

		if !ok {
			return nil, fmt.Errorf("shape.reshape config.shape: must be a sequence")
		}

		target := make([]int, len(raw))

		for idx, val := range raw {
			dimension, err := strictManifestInt(val)

			if err != nil {
				return nil, fmt.Errorf("shape.reshape config.shape[%d]: %w", idx, err)
			}

			target[idx] = dimension
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
	Register("shape.upsample_nearest2d", func(config map[string]any) (operation.Operation, error) {
		return shape.NewUpsampleNearest2D(
			intParamDefault(config, "scale_h", intParamDefault(config, "scale_factor", 0)),
			intParamDefault(config, "scale_w", intParamDefault(config, "scale_factor", 0)),
		), nil
	})
	Register("shape.view_as_heads", func(config map[string]any) (operation.Operation, error) {
		return shape.NewViewAsHeads(intParam(config, "num_heads")), nil
	})
	Register("shape.last_token", func(_ map[string]any) (operation.Operation, error) {
		return shape.NewLastToken(), nil
	})
	Register("shape.merge_heads", func(_ map[string]any) (operation.Operation, error) {
		return shape.NewMergeHeads(), nil
	})
}

func registerData() {
	Register("data.huggingface", func(config map[string]any) (operation.Operation, error) {
		dataset, _ := config["dataset"].(string)
		datasetConfig, _ := config["config"].(string)

		if datasetConfig == "" {
			datasetConfig = "default"
		}

		split, _ := config["split"].(string)

		if split == "" {
			split = "train"
		}

		field, _ := config["field"].(string)

		if field == "" {
			field = "label"
		}

		page := intParamDefault(config, "page", 100)

		return data.NewHuggingFace(dataset, datasetConfig, split, field, page), nil
	})
}

func registerTokenizer() {
	Register("tokenizer.load", func(_ map[string]any) (operation.Operation, error) {
		return tokenizerop.NewLoad(), nil
	})
	Register("tokenizer.encode", func(_ map[string]any) (operation.Operation, error) {
		return tokenizerop.NewEncode(), nil
	})
	Register("tokenizer.decode", func(_ map[string]any) (operation.Operation, error) {
		return tokenizerop.NewDecode(), nil
	})
}

func registerTrain() {
	Register("train.loss.mse", func(_ map[string]any) (operation.Operation, error) {
		return train.NewMSELoss(), nil
	})
	Register("train.loss.cross_entropy", func(_ map[string]any) (operation.Operation, error) {
		return train.NewCrossEntropyLoss(), nil
	})
	Register("train.grad.mse", func(_ map[string]any) (operation.Operation, error) {
		return train.NewMSEGrad(), nil
	})
	Register("train.grad.cross_entropy", func(_ map[string]any) (operation.Operation, error) {
		return train.NewCrossEntropyGrad(), nil
	})
	Register("train.optimizer.adam", func(config map[string]any) (operation.Operation, error) {
		return train.NewAdamStep(
			floatParamDefault(config, "lr", 1e-3),
			floatParamDefault(config, "beta1", 0.9),
			floatParamDefault(config, "beta2", 0.999),
			floatParamDefault(config, "eps", 1e-8),
			floatParamDefault(config, "wd", 0),
		), nil
	})
	Register("train.optimizer.adamw", func(config map[string]any) (operation.Operation, error) {
		return train.NewAdamWStep(
			floatParamDefault(config, "lr", 1e-3),
			floatParamDefault(config, "beta1", 0.9),
			floatParamDefault(config, "beta2", 0.999),
			floatParamDefault(config, "eps", 1e-8),
			floatParamDefault(config, "wd", 1e-2),
		), nil
	})
	Register("train.optimizer.adamax", func(config map[string]any) (operation.Operation, error) {
		return train.NewAdaMaxStep(
			floatParamDefault(config, "lr", 2e-3),
			floatParamDefault(config, "beta1", 0.9),
			floatParamDefault(config, "beta2", 0.999),
			floatParamDefault(config, "eps", 1e-8),
		), nil
	})
	Register("train.optimizer.sgd", func(config map[string]any) (operation.Operation, error) {
		return train.NewSGDStep(
			floatParamDefault(config, "lr", 1e-2),
			floatParamDefault(config, "momentum", 0),
			floatParamDefault(config, "wd", 0),
			boolParam(config, "nesterov"),
		), nil
	})
	Register("train.optimizer.lion", func(config map[string]any) (operation.Operation, error) {
		return train.NewLionStep(
			floatParamDefault(config, "lr", 1e-4),
			floatParamDefault(config, "beta1", 0.9),
			floatParamDefault(config, "beta2", 0.99),
			floatParamDefault(config, "wd", 0),
		), nil
	})
	Register("train.optimizer.rmsprop", func(config map[string]any) (operation.Operation, error) {
		return train.NewRMSPropStep(
			floatParamDefault(config, "lr", 1e-2),
			floatParamDefault(config, "alpha", 0.99),
			floatParamDefault(config, "eps", 1e-8),
			floatParamDefault(config, "momentum", 0),
			floatParamDefault(config, "wd", 0),
		), nil
	})
	Register("train.optimizer.hebbian", func(config map[string]any) (operation.Operation, error) {
		return train.NewHebbianStep(
			floatParamDefault(config, "lr", 1e-3),
			floatParamDefault(config, "max_norm", 0),
		), nil
	})
	Register("train.optimizer.lars", func(config map[string]any) (operation.Operation, error) {
		return train.NewLARSStep(
			floatParamDefault(config, "lr", 1e-2),
			floatParamDefault(config, "eta", 1e-3),
			floatParamDefault(config, "momentum", 0.9),
			floatParamDefault(config, "wd", 0),
			floatParamDefault(config, "eps", 1e-8),
		), nil
	})
	Register("train.optimizer.lamb", func(config map[string]any) (operation.Operation, error) {
		return train.NewLAMBStep(
			floatParamDefault(config, "lr", 1e-3),
			floatParamDefault(config, "beta1", 0.9),
			floatParamDefault(config, "beta2", 0.999),
			floatParamDefault(config, "eps", 1e-6),
			floatParamDefault(config, "wd", 0),
		), nil
	})
	Register("train.optimizer.adagrad", func(config map[string]any) (operation.Operation, error) {
		return train.NewAdaGradStep(
			floatParamDefault(config, "lr", 1e-2),
			floatParamDefault(config, "eps", 1e-10),
			floatParamDefault(config, "wd", 0),
			floatParamDefault(config, "lr_decay", 0),
		), nil
	})
	Register("train.optimizer.adadelta", func(config map[string]any) (operation.Operation, error) {
		return train.NewAdaDeltaStep(
			floatParamDefault(config, "rho", 0.9),
			floatParamDefault(config, "eps", 1e-6),
			floatParamDefault(config, "wd", 0),
		), nil
	})
	Register("train.optimizer.lbfgs", func(config map[string]any) (operation.Operation, error) {
		return train.NewLBFGSStep(
			floatParamDefault(config, "lr", 1.0),
			intParamDefault(config, "hist_size", 10),
			boolParam(config, "line_search"),
			floatParamDefault(config, "c1", 1e-4),
		), nil
	})

	for aliasID, targetID := range map[string]string{
		"optimizer.adam":     "train.optimizer.adam",
		"optimizer.adamw":    "train.optimizer.adamw",
		"optimizer.adamax":   "train.optimizer.adamax",
		"optimizer.sgd":      "train.optimizer.sgd",
		"optimizer.lion":     "train.optimizer.lion",
		"optimizer.rmsprop":  "train.optimizer.rmsprop",
		"optimizer.hebbian":  "train.optimizer.hebbian",
		"optimizer.lars":     "train.optimizer.lars",
		"optimizer.lamb":     "train.optimizer.lamb",
		"optimizer.adagrad":  "train.optimizer.adagrad",
		"optimizer.adadelta": "train.optimizer.adadelta",
		"optimizer.lbfgs":    "train.optimizer.lbfgs",
	} {
		targetID := targetID

		Register(aliasID, func(config map[string]any) (operation.Operation, error) {
			return Build(targetID, config)
		})
	}

	Register("train.checkpoint.save", func(config map[string]any) (operation.Operation, error) {
		dir, _ := config["dir"].(string)
		prefix, _ := config["prefix"].(string)

		return train.NewCheckpointSave(dir, prefix), nil
	})
	Register("train.checkpoint.load", func(config map[string]any) (operation.Operation, error) {
		path, _ := config["path"].(string)

		return train.NewCheckpointLoad(path), nil
	})
}

func registerBench() {
	Register("bench.metric.accuracy", func(_ map[string]any) (operation.Operation, error) {
		return bench.NewAccuracy(), nil
	})
	Register("bench.metric.perplexity", func(_ map[string]any) (operation.Operation, error) {
		return bench.NewPerplexity(), nil
	})
	Register("bench.metric.f1", func(_ map[string]any) (operation.Operation, error) {
		return bench.NewF1(), nil
	})
}

func floatParamDefault(config map[string]any, key string, defaultVal float64) float64 {
	val, ok := config[key]

	if !ok {
		return defaultVal
	}

	f, ok := val.(float64)

	if !ok {
		return defaultVal
	}

	return f
}

func intParam(config map[string]any, key string) int {
	return anyToInt(config[key])
}

func requiredIntParam(config map[string]any, key string) (int, error) {
	value, ok := config[key]

	if !ok {
		return 0, fmt.Errorf("missing required integer")
	}

	return strictManifestInt(value)
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

func registerModel() {
	Register("model.load", func(config map[string]any) (operation.Operation, error) {
		source, _ := config["source"].(string)
		file, _ := config["file"].(string)
		cache, _ := config["cache"].(string)

		return model.NewLoader(source, file, cache), nil
	})
	Register("model.surgery", func(config map[string]any) (operation.Operation, error) {
		source, _ := config["source"].(string)
		op, _ := config["op"].(string)
		at, _ := config["at"].(string)
		after, _ := config["after"].(string)
		name, _ := config["name"].(string)

		var layer []float64

		if raw, ok := config["layer"].([]any); ok {
			layer = make([]float64, len(raw))

			for idx, v := range raw {
				layer[idx] = floatFromAny(v)
			}
		}

		return model.NewSurgery(source, op, at, after, name, layer), nil
	})
	Register("model.graft", func(config map[string]any) (operation.Operation, error) {
		source, _ := config["source"].(string)
		at, _ := config["at"].(string)
		mode, _ := config["mode"].(string)

		return model.NewGraft(source, at, mode), nil
	})
	Register("model.lora", func(config map[string]any) (operation.Operation, error) {
		source, _ := config["source"].(string)
		preset, _ := config["preset"].(string)
		rank := intParamDefault(config, "rank", 8)
		alpha := floatParamDefault(config, "alpha", 0)

		var targets []string

		if raw, ok := config["targets"].([]any); ok {
			for _, v := range raw {
				if s, ok := v.(string); ok {
					targets = append(targets, s)
				}
			}
		}

		return model.NewLoRA(source, preset, targets, rank, alpha), nil
	})
	Register("model.adapter", func(config map[string]any) (operation.Operation, error) {
		source, _ := config["source"].(string)
		at, _ := config["at"].(string)
		reduction := intParamDefault(config, "reduction", 16)

		return model.NewAdapter(source, at, reduction), nil
	})
	Register("model.freeze", func(config map[string]any) (operation.Operation, error) {
		source, _ := config["source"].(string)
		pattern, _ := config["pattern"].(string)
		except, _ := config["except"].(string)
		frozen := boolParamDefault(config, "frozen", true)

		return model.NewFreeze(source, pattern, except, frozen), nil
	})
}

func registerVSA() {
	Register("vsa.bind", func(_ map[string]any) (operation.Operation, error) {
		return vsa.NewBind(), nil
	})
	Register("vsa.bundle", func(_ map[string]any) (operation.Operation, error) {
		return vsa.NewBundle(), nil
	})
	Register("vsa.similarity", func(_ map[string]any) (operation.Operation, error) {
		return vsa.NewSimilarity(), nil
	})
	Register("vsa.permute", func(config map[string]any) (operation.Operation, error) {
		return vsa.NewPermute(intParam(config, "k")), nil
	})
	Register("vsa.inverse_permute", func(config map[string]any) (operation.Operation, error) {
		return vsa.NewInversePermute(intParam(config, "k")), nil
	})
}

func registerPredictiveCoding() {
	Register("predictive_coding.prediction", func(_ map[string]any) (operation.Operation, error) {
		return predictive_coding.NewPrediction(), nil
	})
	Register("predictive_coding.prediction_error", func(_ map[string]any) (operation.Operation, error) {
		return predictive_coding.NewPredictionError(), nil
	})
	Register("predictive_coding.update_representation", func(_ map[string]any) (operation.Operation, error) {
		return predictive_coding.NewUpdateRepresentation(), nil
	})
	Register("predictive_coding.update_weights", func(_ map[string]any) (operation.Operation, error) {
		return predictive_coding.NewUpdateWeights(), nil
	})
}

func registerActiveInference() {
	Register("active_inference.free_energy", func(_ map[string]any) (operation.Operation, error) {
		return active_inference.NewFreeEnergy(), nil
	})
	Register("active_inference.belief_update", func(_ map[string]any) (operation.Operation, error) {
		return active_inference.NewBeliefUpdate(), nil
	})
	Register("active_inference.precision_weight", func(_ map[string]any) (operation.Operation, error) {
		return active_inference.NewPrecisionWeight(), nil
	})
	Register("active_inference.expected_free_energy", func(_ map[string]any) (operation.Operation, error) {
		return active_inference.NewExpectedFreeEnergy(), nil
	})
}

func registerMarkovBlanket() {
	Register("markov_blanket.partition", func(_ map[string]any) (operation.Operation, error) {
		return markov_blanket.NewPartition(), nil
	})
	Register("markov_blanket.flow_internal", func(_ map[string]any) (operation.Operation, error) {
		return markov_blanket.NewFlowInternal(), nil
	})
	Register("markov_blanket.flow_active", func(_ map[string]any) (operation.Operation, error) {
		return markov_blanket.NewFlowActive(), nil
	})
	Register("markov_blanket.mutual_information", func(_ map[string]any) (operation.Operation, error) {
		return markov_blanket.NewMutualInformation(), nil
	})
}

func registerHawkes() {
	Register("hawkes.intensity", func(_ map[string]any) (operation.Operation, error) {
		return hawkes.NewIntensity(), nil
	})
	Register("hawkes.kernel_matrix", func(_ map[string]any) (operation.Operation, error) {
		return hawkes.NewKernelMatrix(), nil
	})
	Register("hawkes.log_likelihood", func(_ map[string]any) (operation.Operation, error) {
		return hawkes.NewLogLikelihood(), nil
	})
	Register("hawkes.simulate", func(_ map[string]any) (operation.Operation, error) {
		return hawkes.NewSimulate(), nil
	})
}

func registerCausal() {
	Register("causal.do_calculus", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewDoCalculus(), nil
	})
	Register("causal.backdoor_adjustment", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewBackdoorAdjustment(), nil
	})
	Register("causal.frontdoor_adjustment", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewFrontdoorAdjustment(), nil
	})
	Register("causal.counterfactual", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewCounterfactual(), nil
	})
	Register("causal.iv_estimate", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewIVEstimate(), nil
	})
	Register("causal.cate", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewCATE(), nil
	})
	Register("causal.dag_markov_factorization", func(_ map[string]any) (operation.Operation, error) {
		return causal.NewDAGMarkovFactorization(), nil
	})
}

func floatFromAny(v any) float64 {
	switch cast := v.(type) {
	case float64:
		return cast
	case int:
		return float64(cast)
	default:
		return 0
	}
}

func strictManifestInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		if typed != float64(int(typed)) {
			return 0, fmt.Errorf("must be an integer, got %v", typed)
		}

		return int(typed), nil
	case string:
		parsed, err := strconv.Atoi(typed)

		if err != nil {
			return 0, fmt.Errorf("must be an integer, got %q", typed)
		}

		return parsed, nil
	default:
		return 0, fmt.Errorf("must be an integer, got %T", value)
	}
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
