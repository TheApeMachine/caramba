package runtime

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	puterdevice "github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/metal"
)

var errDeviceDispatchUnsupported = errors.New("metal graph runner: operation not dispatched via device backend")

/*
evaluateNodeDevice runs IR nodes through device.Backend so every dtype uses the
same batched Metal command stream (unsafe.Pointer + count + dtype.DType).
*/
func (runner *MetalGraphRunner) evaluateNodeDevice(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	operationID := node.OperationID()

	if operationID == "" {
		operationID = ir.OpID(node.OpType())
	}

	switch operationID {
	case "math.add":
		return runner.deviceBinary(node, values, weightsPath, runner.device.Add)
	case "math.mul":
		return runner.deviceBinary(node, values, weightsPath, runner.device.Mul)
	case "math.sub":
		return runner.deviceBinary(node, values, weightsPath, runner.device.Sub)
	case "math.rmsnorm":
		return runner.deviceRMSNorm(node, values, weightsPath)
	case "math.groupnorm":
		return runner.deviceGroupNorm(node, values, weightsPath)
	case "math.adaptive_rmsnorm":
		return runner.deviceAdaptiveRMSNorm(node, values, weightsPath)
	case "embedding.token":
		return runner.deviceEmbedding(node, values, weightsPath)
	case "embedding.timestep":
		return runner.deviceTimestepEmbedding(node, values, weightsPath)
	case "activation.swiglu":
		return runner.deviceSwiGLU(node, values, weightsPath)
	default:
		return nil, nil, errDeviceDispatchUnsupported
	}
}

func (runner *MetalGraphRunner) deviceAdaptiveRMSNorm(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	_ = weightsPath

	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	if len(nodeInputs) != 2 {
		return nil, nil, fmt.Errorf("metal graph runner: %q expects 2 inputs", node.ID())
	}

	output, err := runner.newOutputTensor(node, nodeInputs)

	if err != nil {
		return nil, nil, err
	}

	if err := runner.memory.AdaptiveRMSNorm(nodeInputs[0], nodeInputs[1], output); err != nil {
		_ = output.Close()

		return nil, nil, err
	}

	return output, nil, nil
}

func (runner *MetalGraphRunner) deviceTimestepEmbedding(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	_ = weightsPath

	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	if len(nodeInputs) != 1 {
		return nil, nil, fmt.Errorf("metal graph runner: %q expects 1 input", node.ID())
	}

	timesteps, err := timestepInputValues(nodeInputs[0])

	if err != nil {
		return nil, nil, err
	}

	embeddingDim := int(int64Attribute(node, "dim"))

	if embeddingDim <= 0 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	outputShape, err := nodeOutputShape(node, []tensor.Shape{nodeInputs[0].Shape()})

	if err != nil {
		return nil, nil, err
	}

	embedding := timestepEmbeddingValues(node, timesteps, embeddingDim)
	output, err := runner.memory.Upload(outputShape, dtype.Float32, convert.Float32ToBytes(embedding))

	if err != nil {
		return nil, nil, err
	}

	return output, nil, nil
}

func (runner *MetalGraphRunner) deviceGroupNorm(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	if len(nodeInputs) != 1 {
		return nil, nil, fmt.Errorf("metal graph runner: %q expects 1 input", node.ID())
	}

	scaleTensor, err := runner.loadNodeWeight(node, weightsPath)

	if err != nil {
		return nil, nil, err
	}

	biasTensor, err := runner.loadNodeBias(node, nodeInputs[0], weightsPath)

	if err != nil {
		return nil, nil, err
	}

	output, err := runner.newOutputTensor(node, nodeInputs)

	if err != nil {
		return nil, nil, err
	}

	inputDims := nodeInputs[0].Shape().Dims()

	if len(inputDims) < 3 || !nodeInputs[0].Shape().Equal(output.Shape()) {
		_ = output.Close()

		return nil, nil, tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	channels := inputDims[1]
	spatial := 1

	for _, dimension := range inputDims[2:] {
		spatial *= dimension
	}

	groups := int(int64Attribute(node, "groups"))

	if groups == 0 {
		groups = 32
	}

	runner.device.GroupNorm(
		puterdevice.GroupNormConfig{Groups: groups},
		metal.Resident(nodeInputs[0]),
		metal.Resident(scaleTensor),
		metal.Resident(biasTensor),
		metal.Resident(output),
		batch,
		channels,
		spatial,
		runner.executionDType(node),
	)

	return output, nil, nil
}

type deviceBinaryFunc func(dst, left, right unsafe.Pointer, count int, format dtype.DType)

func (runner *MetalGraphRunner) deviceBinary(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
	dispatch deviceBinaryFunc,
) (tensor.Tensor, []tensor.Tensor, error) {
	_ = weightsPath

	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	if len(nodeInputs) != 2 {
		return nil, nil, fmt.Errorf("metal graph runner: %q expects 2 inputs", node.ID())
	}

	output, err := runner.newOutputTensor(node, nodeInputs)

	if err != nil {
		return nil, nil, err
	}

	executionDType := runner.executionDType(node)
	count := nodeInputs[0].Len()

	if count != nodeInputs[1].Len() || count != output.Len() {
		return nil, nil, tensor.ErrShapeMismatch
	}

	dispatch(
		metal.Resident(output),
		metal.Resident(nodeInputs[0]),
		metal.Resident(nodeInputs[1]),
		count,
		executionDType,
	)

	return output, nil, nil
}

func (runner *MetalGraphRunner) deviceRMSNorm(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	if len(nodeInputs) != 1 {
		return nil, nil, fmt.Errorf("metal graph runner: %q expects 1 input", node.ID())
	}

	weightTensor, err := runner.rmsNormWeight(node, nodeInputs[0], weightsPath)

	if err != nil {
		return nil, nil, err
	}

	output, err := runner.newOutputTensor(node, nodeInputs)

	if err != nil {
		return nil, nil, err
	}

	inputDims := nodeInputs[0].Shape().Dims()

	if len(inputDims) < 1 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	lastDim := inputDims[len(inputDims)-1]
	rows := nodeInputs[0].Len() / lastDim

	if rows*lastDim != nodeInputs[0].Len() || rows*lastDim != output.Len() {
		return nil, nil, tensor.ErrShapeMismatch
	}

	executionDType := runner.executionDType(node)

	runner.device.RMSNorm(
		metal.Resident(nodeInputs[0]),
		metal.Resident(weightTensor),
		metal.Resident(output),
		rows,
		lastDim,
		executionDType,
	)

	return output, nil, nil
}

func (runner *MetalGraphRunner) rmsNormWeight(
	node *ir.Node,
	input tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, error) {
	if isAffineFree(node) {
		return runner.unitRMSNormWeight(input.Shape(), runner.executionDType(node))
	}

	return runner.loadNodeWeight(node, weightsPath)
}

func isAffineFree(node *ir.Node) bool {
	attribute := node.Attribute("affine")

	return attribute.Value == "false"
}

func (runner *MetalGraphRunner) unitRMSNormWeight(
	inputShape tensor.Shape,
	format dtype.DType,
) (tensor.Tensor, error) {
	inputDims := inputShape.Dims()

	if len(inputDims) == 0 {
		return nil, tensor.ErrShapeMismatch
	}

	width := inputDims[len(inputDims)-1]
	shape, err := tensor.NewShape([]int{width})

	if err != nil {
		return nil, err
	}

	values := make([]float32, width)

	for valueIndex := range values {
		values[valueIndex] = 1
	}

	return runner.memory.Upload(shape, format, unitFloatBytes(values, format))
}

func unitFloatBytes(values []float32, format dtype.DType) []byte {
	switch format {
	case dtype.BFloat16:
		var bf16 dtype.BF16

		return bf16.EncodeFloat32(values)
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for valueIndex, value := range values {
			encoded[valueIndex] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded)
	default:
		return convert.Float32ToBytes(values)
	}
}

func (runner *MetalGraphRunner) deviceEmbedding(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	if len(nodeInputs) != 1 {
		return nil, nil, fmt.Errorf("metal graph runner: %q expects 1 input", node.ID())
	}

	weightTensor, err := runner.loadNodeWeight(node, weightsPath)

	if err != nil {
		return nil, nil, err
	}

	output, err := runner.newOutputTensor(node, nodeInputs)

	if err != nil {
		return nil, nil, err
	}

	weightDims := weightTensor.Shape().Dims()

	if len(weightDims) < 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	vocab := weightDims[0]
	hidden := weightDims[1]
	indexCount := nodeInputs[0].Len()
	executionDType := runner.executionDType(node)

	runner.device.Lookup(
		metal.Resident(weightTensor),
		metal.Resident(nodeInputs[0]),
		metal.Resident(output),
		vocab,
		hidden,
		indexCount,
		executionDType,
	)

	return output, nil, nil
}

func (runner *MetalGraphRunner) deviceSwiGLU(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	_ = weightsPath

	nodeInputs, err := runner.nodeInputs(node, values)

	if err != nil {
		return nil, nil, err
	}

	executionDType := runner.executionDType(node)

	switch len(nodeInputs) {
	case 1:
		inputDims := nodeInputs[0].Shape().Dims()

		if len(inputDims) < 1 {
			return nil, nil, tensor.ErrShapeMismatch
		}

		lastDim := inputDims[len(inputDims)-1]

		if lastDim%2 != 0 {
			return nil, nil, tensor.ErrShapeMismatch
		}

		halfCount := lastDim / 2
		batch := nodeInputs[0].Len() / lastDim
		outputShape, err := packedGLUOutputShape(nodeInputs[0].Shape())

		if err != nil {
			return nil, nil, err
		}

		output, err := runner.memory.NewZeroed(outputShape, executionDType)

		if err != nil {
			return nil, nil, err
		}

		runner.device.SwiGLU(
			metal.Resident(output),
			metal.Resident(nodeInputs[0]),
			batch,
			halfCount,
			executionDType,
		)

		return output, nil, nil
	case 2:
		output, err := runner.newOutputTensor(node, nodeInputs)

		if err != nil {
			return nil, nil, err
		}

		elementCount := nodeInputs[0].Len()

		if elementCount != nodeInputs[1].Len() || elementCount != output.Len() {
			return nil, nil, tensor.ErrShapeMismatch
		}

		runner.device.SwiGLUTensors(
			metal.Resident(output),
			metal.Resident(nodeInputs[0]),
			metal.Resident(nodeInputs[1]),
			elementCount,
			executionDType,
		)

		return output, nil, nil
	default:
		return nil, nil, fmt.Errorf(
			"metal graph runner: %q expects 1 or 2 inputs, got %d",
			node.ID(),
			len(nodeInputs),
		)
	}
}

func packedGLUOutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	inputDims := inputShape.Dims()

	if len(inputDims) < 1 {
		return tensor.Shape{}, tensor.ErrShapeMismatch
	}

	lastIndex := len(inputDims) - 1

	if inputDims[lastIndex]%2 != 0 {
		return tensor.Shape{}, tensor.ErrShapeMismatch
	}

	outputDims := append([]int(nil), inputDims...)
	outputDims[lastIndex] = inputDims[lastIndex] / 2

	return tensor.NewShape(outputDims)
}

func (runner *MetalGraphRunner) nodeInputs(
	node *ir.Node,
	values map[string]tensor.Tensor,
) ([]tensor.Tensor, error) {
	nodeInputs := make([]tensor.Tensor, len(node.Inputs()))

	for index, inputNode := range node.Inputs() {
		value, ok := values[inputNode.ID()]

		if !ok || value == nil {
			return nil, fmt.Errorf("metal graph runner: missing input %q for %q", inputNode.ID(), node.ID())
		}

		nodeInputs[index] = value
	}

	return nodeInputs, nil
}

func (runner *MetalGraphRunner) loadNodeWeight(
	node *ir.Node,
	weightsPath string,
) (tensor.Tensor, error) {
	weightName, ok := node.Metadata()["weight_name"].(string)

	if !ok || weightName == "" || weightsPath == "" {
		return nil, fmt.Errorf("metal graph runner: node %q has no weight", node.ID())
	}

	if sliceAxis, start, end, ok := nodeWeightSlice(node, weightName); ok {
		return runner.loadWeightTensorSlice(
			weightPathForNode(node, weightsPath),
			weightName,
			runner.executionDType(node),
			sliceAxis,
			start,
			end,
		)
	}

	return runner.loadWeightTensor(
		weightPathForNode(node, weightsPath),
		weightName,
		runner.executionDType(node),
		node.Shape(),
	)
}

func (runner *MetalGraphRunner) loadNodeBias(
	node *ir.Node,
	input tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, error) {
	weightName, ok := node.Metadata()["weight_name"].(string)

	if !ok || weightName == "" || weightsPath == "" {
		return nil, fmt.Errorf("metal graph runner: node %q has no bias", node.ID())
	}

	biasName := biasNameForWeight(weightName)
	if biasName == "" {
		return nil, fmt.Errorf("metal graph runner: node %q cannot infer bias from %q", node.ID(), weightName)
	}

	inputDims := input.Shape().Dims()

	if len(inputDims) < 2 {
		return nil, tensor.ErrShapeMismatch
	}

	fallbackShape, err := tensor.NewShape([]int{inputDims[1]})

	if err != nil {
		return nil, err
	}

	return runner.loadWeightTensor(
		weightPathForNode(node, weightsPath),
		biasName,
		runner.executionDType(node),
		fallbackShape,
	)
}

func (runner *MetalGraphRunner) loadOptionalNodeBias(
	node *ir.Node,
	input tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, error) {
	biasTensor, err := runner.loadNodeBias(node, input, weightsPath)

	if err == nil {
		return biasTensor, nil
	}

	if strings.Contains(err.Error(), "not found") ||
		strings.Contains(err.Error(), "has no bias") ||
		strings.Contains(err.Error(), "cannot infer bias") {
		return nil, nil
	}

	return nil, err
}

func biasNameForWeight(weightName string) string {
	if strings.HasSuffix(weightName, ".weight") {
		return strings.TrimSuffix(weightName, ".weight") + ".bias"
	}

	return ""
}

func nodeWeightSlice(node *ir.Node, weightName string) (string, int64, int64, bool) {
	if sliceAxis, ok := node.Metadata()["weight_slice_axis"].(string); ok && sliceAxis != "" {
		start := int64Metadata(node, "weight_slice_start")

		return sliceAxis, start, weightSliceEnd(node, sliceAxis), true
	}

	if !strings.Contains(weightName, ".attn.to_qkv_mlp_proj.weight") {
		return "", 0, 0, false
	}

	hiddenSize := int64Attribute(node, "in_features")

	if hiddenSize == 0 {
		return "", 0, 0, false
	}

	switch {
	case strings.HasSuffix(node.ID(), ".attn.to_q"):
		return "output", 0, hiddenSize, true
	case strings.HasSuffix(node.ID(), ".attn.to_k"):
		return "output", hiddenSize, hiddenSize * 2, true
	case strings.HasSuffix(node.ID(), ".attn.to_v"):
		return "output", hiddenSize * 2, hiddenSize * 3, true
	case strings.HasSuffix(node.ID(), ".proj_mlp"):
		return "output", hiddenSize * 3, hiddenSize*3 + int64Attribute(node, "out_features"), true
	default:
		return "", 0, 0, false
	}
}

func weightSliceEnd(node *ir.Node, sliceAxis string) int64 {
	if end := int64Metadata(node, "weight_slice_end"); end > 0 {
		return end
	}

	start := int64Metadata(node, "weight_slice_start")

	if sliceAxis == "input" {
		return start + int64Attribute(node, "in_features")
	}

	return start + int64Attribute(node, "out_features")
}

func int64Metadata(node *ir.Node, key string) int64 {
	switch typed := node.Metadata()[key].(type) {
	case int:
		return int64(typed)
	case int64:
		return typed
	case float64:
		return int64(typed)
	default:
		return 0
	}
}

func int64Attribute(node *ir.Node, key string) int64 {
	value, err := strconv.ParseInt(node.Attribute(key).Value, 10, 64)

	if err != nil {
		return 0
	}

	return value
}

func (runner *MetalGraphRunner) executionDType(node *ir.Node) dtype.DType {
	executionDType := node.ValueType().DType

	if executionDType != dtype.Invalid {
		return executionDType
	}

	return dtype.BFloat16
}

func (runner *MetalGraphRunner) newOutputTensor(
	node *ir.Node,
	nodeInputs []tensor.Tensor,
) (tensor.Tensor, error) {
	outputDType := runner.executionDType(node)
	inputShapes := make([]tensor.Shape, 0, len(nodeInputs))

	for _, nodeInput := range nodeInputs {
		inputShapes = append(inputShapes, nodeInput.Shape())
	}

	outputShape, err := nodeOutputShape(node, inputShapes)

	if err != nil {
		return nil, err
	}

	return runner.memory.NewZeroed(outputShape, outputDType)
}

func nodeOutputShape(
	node *ir.Node,
	inputShapes []tensor.Shape,
) (tensor.Shape, error) {
	outputShape := node.Shape()

	if len(inputShapes) > 0 {
		inputDims := inputShapes[0].Dims()
		seqLen := inputDims[0]

		if node.OperationID() == ir.OpEmbeddingToken {
			seqLen = inputShapes[0].Len()
		}

		if node.OperationID() == "embedding.timestep" {
			embeddingDim := int(int64Attribute(node, "dim"))

			if embeddingDim <= 0 {
				return tensor.Shape{}, tensor.ErrShapeMismatch
			}

			return tensor.NewShape([]int{inputShapes[0].Len(), embeddingDim})
		}

		if node.OperationID() == "projection.linear" && len(inputDims) > 0 {
			outputDims := outputShape.Dims()

			if len(outputDims) > 0 {
				newDims := make([]int, len(inputDims))
				copy(newDims, inputDims)
				newDims[len(newDims)-1] = outputDims[len(outputDims)-1]
				return tensor.NewShape(newDims)
			}
		}

		if node.OperationID() == "convolution.conv2d" && len(inputShapes) > 1 {
			return conv2DOutputShape(node, inputShapes[0], inputShapes[1])
		}

		if node.OperationID() == "shape.concat" {
			return concatOutputShape(node, inputShapes)
		}

		if node.OperationID() == "shape.slice" {
			return sliceOutputShape(node, inputShapes[0])
		}

		if node.OperationID() == "shape.reshape" {
			return reshapeOutputShape(node)
		}

		if node.OperationID() == "shape.transpose" {
			return transposeOutputShape(node, inputShapes[0])
		}

		if node.OperationID() == "shape.upsample_nearest2d" {
			return upsampleNearest2DOutputShape(node, inputShapes[0])
		}

		if node.OperationID() == "shape.view_as_heads" {
			return viewAsHeadsOutputShape(node, inputShapes[0])
		}

		if node.OperationID() == "shape.merge_heads" {
			return mergeHeadsOutputShape(node, inputShapes[0])
		}

		if node.OperationID() == "math.adaptive_rmsnorm" {
			return adaptiveRMSNormOutputShape(inputShapes)
		}

		if preservesInputShape(node.OperationID()) {
			return inputShapes[0], nil
		}

		dims := outputShape.Dims()

		if len(dims) > 0 && dims[0] != seqLen {
			newDims := make([]int, len(dims))
			copy(newDims, dims)
			newDims[0] = seqLen
			outputShape, _ = tensor.NewShape(newDims)
		}
	}

	return outputShape, nil
}

func adaptiveRMSNormOutputShape(inputShapes []tensor.Shape) (tensor.Shape, error) {
	if len(inputShapes) != 2 {
		return tensor.Shape{}, tensor.ErrShapeMismatch
	}

	inputDims := inputShapes[0].Dims()
	modulationDims := inputShapes[1].Dims()

	if len(inputDims) < 2 || len(modulationDims) != 2 {
		return tensor.Shape{}, tensor.ErrShapeMismatch
	}

	channelCount := inputDims[len(inputDims)-1]

	if modulationDims[0] != inputDims[0] || modulationDims[1] != 2*channelCount {
		return tensor.Shape{}, tensor.ErrShapeMismatch
	}

	return inputShapes[0], nil
}

func timestepInputValues(input tensor.Tensor) ([]float32, error) {
	storageDType, rawBytes, err := input.RawBytes()

	if err != nil {
		return nil, err
	}

	if storageDType == dtype.Float32 {
		return convert.BytesToFloat32(dtype.Float32, rawBytes)
	}

	if storageDType != dtype.Int32 {
		return nil, fmt.Errorf("metal graph runner: timestep dtype %s is unsupported", storageDType)
	}

	if len(rawBytes)%4 != 0 {
		return nil, tensor.ErrShapeMismatch
	}

	values := make([]float32, len(rawBytes)/4)

	for index := range values {
		values[index] = float32(int32(binary.LittleEndian.Uint32(rawBytes[index*4:])))
	}

	return values, nil
}

func timestepEmbeddingValues(
	node *ir.Node,
	timesteps []float32,
	embeddingDim int,
) []float32 {
	halfDim := embeddingDim / 2
	denominator := float32(halfDim) - float32Attribute(node, "downscale_freq_shift", 1)
	maxPeriod := float32Attribute(node, "max_period", 10000)
	scale := float32Attribute(node, "scale", 1)
	flipSinToCos := boolAttribute(node, "flip_sin_to_cos")
	output := make([]float32, len(timesteps)*embeddingDim)

	for timestepIndex, timestep := range timesteps {
		rowOffset := timestepIndex * embeddingDim

		for channelIndex := range halfDim {
			exponent := -float32(math.Log(float64(maxPeriod))) * float32(channelIndex) / denominator
			phase := timestep * float32(math.Exp(float64(exponent))) * scale
			sinValue := float32(math.Sin(float64(phase)))
			cosValue := float32(math.Cos(float64(phase)))

			if flipSinToCos {
				output[rowOffset+channelIndex] = cosValue
				output[rowOffset+halfDim+channelIndex] = sinValue

				continue
			}

			output[rowOffset+channelIndex] = sinValue
			output[rowOffset+halfDim+channelIndex] = cosValue
		}
	}

	return output
}

func float32Attribute(node *ir.Node, name string, defaultValue float32) float32 {
	value, ok := node.Attributes()[name]

	if !ok {
		return defaultValue
	}

	parsed, err := strconv.ParseFloat(value.Value, 32)

	if err != nil {
		return defaultValue
	}

	return float32(parsed)
}

func boolAttribute(node *ir.Node, name string) bool {
	value, ok := node.Attributes()[name]

	if !ok {
		return false
	}

	parsed, err := strconv.ParseBool(value.Value)

	if err != nil {
		return false
	}

	return parsed
}

func conv2DOutputShape(
	node *ir.Node,
	inputShape tensor.Shape,
	weightShape tensor.Shape,
) (tensor.Shape, error) {
	inputDims := inputShape.Dims()
	weightDims := weightShape.Dims()

	if len(inputDims) != 4 || len(weightDims) != 4 {
		return node.Shape(), nil
	}

	strideH := positiveIntAttribute(node, "stride_h", 1)
	strideW := positiveIntAttribute(node, "stride_w", 1)
	padH := int(int64Attribute(node, "pad_h"))
	padW := int(int64Attribute(node, "pad_w"))
	outHeight := (inputDims[2]+2*padH-weightDims[2])/strideH + 1
	outWidth := (inputDims[3]+2*padW-weightDims[3])/strideW + 1

	return tensor.NewShape([]int{
		inputDims[0],
		weightDims[0],
		outHeight,
		outWidth,
	})
}

func positiveIntAttribute(node *ir.Node, key string, fallback int) int {
	value := int(int64Attribute(node, key))

	if value <= 0 {
		return fallback
	}

	return value
}

func upsampleNearest2DOutputShape(
	node *ir.Node,
	inputShape tensor.Shape,
) (tensor.Shape, error) {
	inputDims := inputShape.Dims()

	if len(inputDims) != 4 {
		return node.Shape(), nil
	}

	scaleH := int(int64Attribute(node, "scale_h"))
	scaleW := int(int64Attribute(node, "scale_w"))

	if scaleH <= 0 || scaleW <= 0 {
		return node.Shape(), nil
	}

	return tensor.NewShape([]int{
		inputDims[0],
		inputDims[1],
		inputDims[2] * scaleH,
		inputDims[3] * scaleW,
	})
}

func reshapeOutputShape(node *ir.Node) (tensor.Shape, error) {
	attribute := node.Attribute("shape")

	if attribute.Value == "" {
		return node.Shape(), nil
	}

	fields := strings.Fields(strings.Trim(attribute.Value, "[]"))
	dims := make([]int, 0, len(fields))

	for _, field := range fields {
		dimension, err := strconv.ParseInt(strings.Trim(field, ","), 10, 32)

		if err != nil {
			return node.Shape(), nil
		}

		dims = append(dims, int(dimension))
	}

	if len(dims) == 0 {
		return node.Shape(), nil
	}

	return tensor.NewShape(dims)
}

func transposeOutputShape(
	node *ir.Node,
	inputShape tensor.Shape,
) (tensor.Shape, error) {
	outputDims := inputShape.Dims()
	dim0 := int(int64Attribute(node, "dim0"))
	dim1 := int(int64Attribute(node, "dim1"))

	if len(outputDims) == 0 || dim0 < 0 || dim0 >= len(outputDims) || dim1 < 0 || dim1 >= len(outputDims) {
		return node.Shape(), nil
	}

	outputDims[dim0], outputDims[dim1] = outputDims[dim1], outputDims[dim0]

	return tensor.NewShape(outputDims)
}

func sliceOutputShape(
	node *ir.Node,
	inputShape tensor.Shape,
) (tensor.Shape, error) {
	inputDims := inputShape.Dims()
	axis := concatAxis(node, len(inputDims))
	start := int(int64Attribute(node, "start"))
	end := int(int64Attribute(node, "end"))

	if len(inputDims) == 0 || start < 0 || end <= start || end > inputDims[axis] {
		return node.Shape(), nil
	}

	outputDims := append([]int(nil), inputDims...)
	outputDims[axis] = end - start

	return tensor.NewShape(outputDims)
}

func viewAsHeadsOutputShape(
	node *ir.Node,
	inputShape tensor.Shape,
) (tensor.Shape, error) {
	inputDims := inputShape.Dims()
	outputDims := node.Shape().Dims()

	if len(inputDims) != 3 || len(outputDims) != 4 {
		return node.Shape(), nil
	}

	return tensor.NewShape([]int{
		inputDims[0],
		inputDims[1],
		outputDims[2],
		outputDims[3],
	})
}

func mergeHeadsOutputShape(
	node *ir.Node,
	inputShape tensor.Shape,
) (tensor.Shape, error) {
	inputDims := inputShape.Dims()

	if len(inputDims) != 4 {
		return node.Shape(), nil
	}

	return tensor.NewShape([]int{
		inputDims[0],
		inputDims[1],
		inputDims[2] * inputDims[3],
	})
}

func preservesInputShape(operationID ir.OpID) bool {
	switch operationID {
	case "math.rmsnorm", "math.layernorm", "math.groupnorm", "positional.rope", "attention.sdpa",
		"math.add", "math.sub", "math.mul",
		"activation.silu", "activation.swish":
		return true
	default:
		return false
	}
}

func concatOutputShape(
	node *ir.Node,
	inputShapes []tensor.Shape,
) (tensor.Shape, error) {
	if len(inputShapes) == 0 {
		return node.Shape(), nil
	}

	outputDims := inputShapes[0].Dims()
	axis := concatAxis(node, len(outputDims))

	for _, inputShape := range inputShapes[1:] {
		inputDims := inputShape.Dims()

		if len(inputDims) != len(outputDims) {
			return tensor.Shape{}, tensor.ErrShapeMismatch
		}

		outputDims[axis] += inputDims[axis]
	}

	return tensor.NewShape(outputDims)
}

func concatAxis(node *ir.Node, rank int) int {
	axis := 0
	attribute := node.Attribute("dim")

	if attribute.Value != "" {
		parsed, err := strconv.ParseInt(attribute.Value, 10, 32)

		if err == nil {
			axis = int(parsed)
		}
	}

	if axis < 0 {
		axis += rank
	}

	if axis < 0 || axis >= rank {
		return 0
	}

	return axis
}
