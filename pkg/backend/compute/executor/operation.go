package executor

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func RunOperation(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	operation state.Operation,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(backend, inputs)
	if err != nil {
		return nil, err
	}

	stateDict := OperationState(node, inputs, values)
	outputState, err := operation.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return UploadOutput(backend, node, inputs, outputState.Out)
}

func RunOptimizer(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	optimizer state.Optimizer,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(backend, inputs)
	if err != nil {
		return nil, err
	}

	stateDict := OperationState(node, inputs, values)
	if len(stateDict.Params) == 0 && len(values) > 0 {
		stateDict.Params = values[0]
	}

	if len(stateDict.Grads) == 0 && len(values) > 1 {
		stateDict.Grads = values[1]
	}

	outputState, err := optimizer.Step(stateDict)

	if err != nil {
		return nil, err
	}

	return UploadOutput(backend, node, inputs, outputState.Out)
}

func OperationState(
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	values [][]float64,
) *state.Dict {
	stateDict := OperationConfig(node).
		WithShape(OperationShape(node, inputs))

	operationInputs := make([]any, len(values))

	for index := range values {
		operationInputs[index] = values[index]
	}

	if len(operationInputs) > 0 {
		stateDict.WithInputs(operationInputs...)
	}

	return stateDict
}

func OperationConfig(node NodeSpec) *state.Dict {
	stateDict := state.NewDict()
	stateDict.NodeID = node.ID

	applyStateMetadata(stateDict, node)

	return stateDict
}

func applyStateMetadata(stateDict *state.Dict, node NodeSpec) {
	if cache, ok := node.Metadata["kv_cache"].(*kv.Cache); ok {
		stateDict.KVCache = cache
	}

	stateDict.Source = stringConfig(node, "source", stateDict.Source)
	stateDict.File = stringConfig(node, "file", stateDict.File)
	stateDict.Cache = stringConfig(node, "cache", stateDict.Cache)
	stateDict.Revision = stringConfig(node, "revision", stateDict.Revision)
	stateDict.RepoType = stringConfig(node, "repo_type", stateDict.RepoType)
	stateDict.Text = stringConfig(node, "text", stateDict.Text)
	stateDict.Op = stringConfig(node, "op", stateDict.Op)
	stateDict.At = stringConfig(node, "at", stateDict.At)
	stateDict.After = stringConfig(node, "after", stateDict.After)
	stateDict.Name = stringConfig(node, "name", stateDict.Name)
	stateDict.Preset = stringConfig(node, "preset", stateDict.Preset)
	stateDict.Pattern = stringConfig(node, "pattern", stateDict.Pattern)
	stateDict.Except = stringConfig(node, "except", stateDict.Except)
	stateDict.Mode = stringConfig(node, "mode", stateDict.Mode)
	stateDict.Targets = stringSliceConfig(node, "targets", stateDict.Targets)

	stateDict.LR = floatConfig(node, "lr", stateDict.LR)
	stateDict.Beta1 = floatConfig(node, "beta1", stateDict.Beta1)
	stateDict.Beta2 = floatConfig(node, "beta2", stateDict.Beta2)
	stateDict.Eps = floatConfig(node, "eps", stateDict.Eps)
	stateDict.WD = floatConfig(node, "wd", stateDict.WD)
	stateDict.Momentum = floatConfig(node, "momentum", stateDict.Momentum)
	stateDict.Alpha = floatConfig(node, "alpha", stateDict.Alpha)
	stateDict.Rho = floatConfig(node, "rho", stateDict.Rho)
	stateDict.Eta = floatConfig(node, "eta", stateDict.Eta)
	stateDict.LRDecay = floatConfig(node, "lr_decay", stateDict.LRDecay)
	stateDict.MaxNorm = floatConfig(node, "max_norm", stateDict.MaxNorm)
	stateDict.Tau = floatConfig(node, "tau", stateDict.Tau)
	stateDict.Theta = floatConfig(node, "theta", stateDict.Theta)
	stateDict.Base = floatConfig(node, "base", stateDict.Base)
	stateDict.InitStd = floatConfig(node, "init_std", stateDict.InitStd)
	stateDict.P = floatConfig(node, "p", stateDict.P)
	stateDict.C1 = floatConfig(node, "c1", stateDict.C1)

	stateDict.Nesterov = boolConfig(node, "nesterov", stateDict.Nesterov)
	stateDict.Centered = boolConfig(node, "centered", stateDict.Centered)
	stateDict.LineSearch = boolConfig(node, "line_search", stateDict.LineSearch)
	stateDict.Training = boolConfig(node, "training", stateDict.Training)
	stateDict.Causal = boolConfig(node, "causal", stateDict.Causal)
	stateDict.Frozen = boolConfig(node, "frozen", stateDict.Frozen)
	stateDict.Ceil = boolConfig(node, "ceil", stateDict.Ceil)
	stateDict.CountPad = boolConfig(node, "count_include_pad", stateDict.CountPad)
	stateDict.CountPad = boolConfig(node, "count_pad", stateDict.CountPad)
	stateDict.Truncate = boolConfig(node, "truncate", stateDict.Truncate)
	stateDict.SkipSpecialTokens = boolConfig(
		node,
		"skip_special_tokens",
		stateDict.SkipSpecialTokens,
	)

	stateDict.HistSize = intConfig(node, "hist_size", stateDict.HistSize)
	stateDict.MaxLength = intConfig(node, "max_length", stateDict.MaxLength)
	stateDict.PadTo = intConfig(node, "pad_to", stateDict.PadTo)
	stateDict.PadID = intConfig(node, "pad_id", stateDict.PadID)
	stateDict.Dim = intConfig(node, "dim", stateDict.Dim)
	stateDict.Dim0 = intConfig(node, "dim0", stateDict.Dim0)
	stateDict.Dim1 = intConfig(node, "dim1", stateDict.Dim1)
	stateDict.SplitSize = intConfig(node, "split_size", stateDict.SplitSize)
	stateDict.Window = intConfig(node, "window", stateDict.Window)
	stateDict.NumHeads = intConfig(node, "num_heads", stateDict.NumHeads)
	stateDict.NumKVHeads = intConfig(node, "num_kv_heads", stateDict.NumKVHeads)
	stateDict.HeadDim = intConfig(node, "head_dim", stateDict.HeadDim)
	stateDict.DModel = intConfig(node, "d_model", stateDict.DModel)
	stateDict.VocabSize = intConfig(node, "vocab_size", stateDict.VocabSize)
	stateDict.InFeatures = intConfig(node, "in_features", stateDict.InFeatures)
	stateDict.OutFeatures = intConfig(node, "out_features", stateDict.OutFeatures)
	stateDict.DIn = intConfig(node, "d_in", stateDict.DIn)
	stateDict.DQ = intConfig(node, "d_q", stateDict.DQ)
	stateDict.DK = intConfig(node, "d_k", stateDict.DK)
	stateDict.DV = intConfig(node, "d_v", stateDict.DV)
	stateDict.InChannels = intConfig(node, "in_c", stateDict.InChannels)
	stateDict.OutChannels = intConfig(node, "out_c", stateDict.OutChannels)
	stateDict.InChannels = intConfig(node, "in_channels", stateDict.InChannels)
	stateDict.OutChannels = intConfig(node, "out_channels", stateDict.OutChannels)
	stateDict.KernelSize = intConfig(node, "kernel_size", stateDict.KernelSize)
	stateDict.KernelD = intConfig(node, "k_d", stateDict.KernelD)
	stateDict.KernelH = intConfig(node, "k_h", stateDict.KernelH)
	stateDict.KernelW = intConfig(node, "k_w", stateDict.KernelW)
	stateDict.KernelD = intConfig(node, "kernel_d", stateDict.KernelD)
	stateDict.KernelH = intConfig(node, "kernel_h", stateDict.KernelH)
	stateDict.KernelW = intConfig(node, "kernel_w", stateDict.KernelW)
	stateDict.Stride = intConfig(node, "stride", stateDict.Stride)
	stateDict.StrideD = intConfig(node, "s_d", stateDict.StrideD)
	stateDict.StrideH = intConfig(node, "stride_h", stateDict.StrideH)
	stateDict.StrideW = intConfig(node, "stride_w", stateDict.StrideW)
	stateDict.StrideH = intConfig(node, "s_h", stateDict.StrideH)
	stateDict.StrideW = intConfig(node, "s_w", stateDict.StrideW)
	stateDict.Padding = intConfig(node, "padding", stateDict.Padding)
	stateDict.PadD = intConfig(node, "p_d", stateDict.PadD)
	stateDict.PadH = intConfig(node, "pad_h", stateDict.PadH)
	stateDict.PadW = intConfig(node, "pad_w", stateDict.PadW)
	stateDict.PadH = intConfig(node, "p_h", stateDict.PadH)
	stateDict.PadW = intConfig(node, "p_w", stateDict.PadW)
	stateDict.Dilation = intConfig(node, "dilation", stateDict.Dilation)
	stateDict.DilationD = intConfig(node, "dil_d", stateDict.DilationD)
	stateDict.DilationH = intConfig(node, "dil_h", stateDict.DilationH)
	stateDict.DilationW = intConfig(node, "dil_w", stateDict.DilationW)
	stateDict.DilationH = intConfig(node, "d_h", stateDict.DilationH)
	stateDict.DilationW = intConfig(node, "d_w", stateDict.DilationW)
	stateDict.Groups = intConfig(node, "groups", stateDict.Groups)
	stateDict.OutPadH = intConfig(node, "out_pad_h", stateDict.OutPadH)
	stateDict.OutPadW = intConfig(node, "out_pad_w", stateDict.OutPadW)
	stateDict.OutH = intConfig(node, "out_h", stateDict.OutH)
	stateDict.OutW = intConfig(node, "out_w", stateDict.OutW)
	stateDict.Divisor = intConfig(node, "divisor_override", stateDict.Divisor)
	stateDict.Reduction = intConfig(node, "reduction", stateDict.Reduction)
	stateDict.Rank = intConfig(node, "rank", stateDict.Rank)
	stateDict.K = intConfig(node, "k", stateDict.K)
	stateDict.Step = intConfig(node, "step", stateDict.Step)

	stateDict.Weight = floatSliceConfig(node, "weight", stateDict.Weight)
	stateDict.Bias = floatSliceConfig(node, "bias", stateDict.Bias)
	stateDict.Layer = floatSliceConfig(node, "layer", stateDict.Layer)

	if targetShape := intSliceConfig(node, "target_shape"); len(targetShape) > 0 {
		stateDict.TargetShape = targetShape
	}

	if shape := intSliceConfig(node, "shape"); len(shape) > 0 {
		stateDict.TargetShape = shape
	}
}

func RunErrorOperation(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	operation func(shape []int, data ...[]float64) ([]float64, error),
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(backend, inputs)
	if err != nil {
		return nil, err
	}

	output, err := operation(OutputShape(node, inputs), values...)
	if err != nil {
		return nil, err
	}

	return UploadOutput(backend, node, inputs, output)
}

func RunForwardErrorOperation(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	operation interface {
		Forward(shape []int, data ...[]float64) ([]float64, error)
	},
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(backend, inputs)
	if err != nil {
		return nil, err
	}

	output, err := operation.Forward(OutputShape(node, inputs), values...)
	if err != nil {
		return nil, err
	}

	return UploadOutput(backend, node, inputs, output)
}

func UploadOutput(
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	output []float64,
) (tensor.Float64Tensor, error) {
	shape, err := OutputTensorShape(node, inputs, output)

	if err != nil {
		return nil, err
	}

	adopter, ok := backend.(interface {
		AdoptFloat64(tensor.Shape, []float64) (tensor.Float64Tensor, error)
	})

	if ok {
		return adopter.AdoptFloat64(shape, output)
	}

	return backend.UploadFloat64(shape, output)
}

func InputValues(
	backend Backend,
	inputs []tensor.Float64Tensor,
) ([][]float64, error) {
	if backend == nil {
		return nil, fmt.Errorf("executor: backend is required for input values")
	}

	values := make([][]float64, len(inputs))

	for index, input := range inputs {
		value, err := backend.DownloadFloat64(input)

		if err != nil {
			return nil, err
		}

		values[index] = value
	}

	return values, nil
}

func OutputShape(node NodeSpec, inputs []tensor.Float64Tensor) []int {
	if len(node.Shape) > 0 {
		return append([]int(nil), node.Shape...)
	}

	if len(inputs) == 0 {
		return nil
	}

	return inputs[0].Shape().Dims()
}

func OperationShape(node NodeSpec, inputs []tensor.Float64Tensor) []int {
	if shape := intSliceConfig(node, "op_shape"); len(shape) > 0 {
		return shape
	}

	if len(inputs) > 0 {
		return inputs[0].Shape().Dims()
	}

	return OutputShape(node, inputs)
}

func OutputTensorShape(
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	output []float64,
) (tensor.Shape, error) {
	shapeData := OutputShape(node, inputs)
	if len(shapeData) == 0 {
		shapeData = []int{len(output)}
	}

	shape, err := tensor.NewShape(shapeData)

	if err != nil {
		return tensor.Shape{}, err
	}

	if shape.Len() != len(output) {
		return tensor.Shape{}, fmt.Errorf(
			"executor: %s node %q output has %d values for shape length %d",
			node.Op,
			node.ID,
			len(output),
			shape.Len(),
		)
	}

	return shape, nil
}

func NormalizeOperation(op ir.OpType) ir.OpType {
	switch strings.ToLower(string(op)) {
	case "input", "data.input":
		return ir.OpInput
	case "add", "math.add":
		return ir.OpAdd
	case "mul", "math.mul":
		return ir.OpMul
	case "matmul", "math.matmul":
		return ir.OpMatmul
	case "relu", "activation.relu":
		return ir.OpReLU
	case "leakyrelu", "leaky_relu", "activation.leaky_relu":
		return ir.OpLeakyReLU
	case "gelu", "activation.gelu":
		return ir.OpGELU
	case "tanh", "activation.tanh":
		return ir.OpTanh
	case "sigmoid", "activation.sigmoid":
		return ir.OpSigmoid
	case "swiglu", "activation.swiglu":
		return ir.OpSwiGLU
	case "swish", "activation.swish":
		return ir.OpSwish
	case "selu", "activation.selu":
		return ir.OpSELU
	case "fused", "math.matmul_add", "math.matmul_add_gelu":
		return ir.OpFused
	default:
		return op
	}
}

func intConfig(node NodeSpec, key string, fallback int) int {
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

func floatConfig(node NodeSpec, key string, fallback float64) float64 {
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

func boolConfig(node NodeSpec, key string, fallback bool) bool {
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

func stringConfig(node NodeSpec, key string, fallback string) string {
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

func intSliceConfig(node NodeSpec, key string) []int {
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
	case []any:
		values := make([]int, len(typed))

		for index, item := range typed {
			switch number := item.(type) {
			case int:
				values[index] = number
			case int64:
				values[index] = int(number)
			case float64:
				values[index] = int(number)
			case float32:
				values[index] = int(number)
			}
		}

		return values
	default:
		return nil
	}
}

func floatSliceConfig(node NodeSpec, key string, fallback []float64) []float64 {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case []float64:
		return typed
	case []float32:
		values := make([]float64, len(typed))

		for index, item := range typed {
			values[index] = float64(item)
		}

		return values
	case []any:
		values := make([]float64, len(typed))

		for index, item := range typed {
			switch number := item.(type) {
			case float64:
				values[index] = number
			case float32:
				values[index] = float64(number)
			case int:
				values[index] = float64(number)
			case int64:
				values[index] = float64(number)
			}
		}

		return values
	default:
		return fallback
	}
}

func stringSliceConfig(node NodeSpec, key string, fallback []string) []string {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		values := make([]string, 0, len(typed))

		for _, item := range typed {
			text, ok := item.(string)

			if ok {
				values = append(values, text)
			}
		}

		return values
	default:
		return fallback
	}
}
