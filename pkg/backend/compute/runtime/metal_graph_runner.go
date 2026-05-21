package runtime

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/metal"
	"github.com/theapemachine/puter/kernels"
)

/*
MetalGraphRunner executes lowered IR graphs on Metal through direct kernel dispatch.
*/
type MetalGraphRunner struct {
	memory       *metal.Backend
	device       device.Backend
	weightsCache map[string]tensor.Tensor
}

/*
NewMetalGraphRunner constructs a MetalGraphRunner.
*/
func NewMetalGraphRunner(memory *metal.Backend, deviceBackend device.Backend) *MetalGraphRunner {
	return &MetalGraphRunner{
		memory:       memory,
		device:       deviceBackend,
		weightsCache: make(map[string]tensor.Tensor),
	}
}

func (runner *MetalGraphRunner) Memory() *metal.Backend {
	return runner.memory
}

/*
Execute evaluates all graph nodes and returns requested target outputs.
*/
func (runner *MetalGraphRunner) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
	weightsPath string,
	externalInputs map[string]tensor.Tensor,
) (map[string]tensor.Tensor, error) {
	if graph == nil {
		return nil, fmt.Errorf("metal graph runner: graph is required")
	}

	layers, err := graph.TopologyLayers()

	if err != nil {
		return nil, err
	}

	values := make(map[string]tensor.Tensor, len(graph.Nodes()))

	for key, value := range externalInputs {
		values[key] = value
	}

	runner.memory.BeginBatch()
	batchActive := true
	defer func() {
		if batchActive {
			runner.memory.EndBatch()
		}
	}()

	var temporaries []tensor.Tensor
	defer func() {
		for _, temp := range temporaries {
			_ = temp.Close()
		}
	}()

	for _, layer := range layers {
		for _, node := range layer {
			if err := ctx.Err(); err != nil {
				closeValues(values)
				return nil, err
			}

			value, temps, evalErr := runner.evaluateNode(ctx, node, values, weightsPath)
			temporaries = append(temporaries, temps...)

			if evalErr != nil {
				closeValues(values)
				return nil, evalErr
			}

			if value != nil {
				values[node.ID()] = value
			}
		}
	}

	runner.memory.EndBatch()
	batchActive = false

	outputs := make(map[string]tensor.Tensor, len(targets))

	for _, target := range targets {
		value, ok := values[target.ID()]

		if !ok || value == nil {
			closeValues(values)

			return nil, fmt.Errorf("metal graph runner: target %q produced no output", target.ID())
		}

		outputs[target.ID()] = value
	}

	// Close intermediate tensors that are not in outputs or externalInputs
	for key, value := range values {
		if _, isOutput := outputs[key]; !isOutput {
			if _, isExternal := externalInputs[key]; !isExternal {
				_ = value.Close()
			}
		}
	}

	return outputs, nil
}

func (runner *MetalGraphRunner) evaluateNode(
	ctx context.Context,
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, []tensor.Tensor, error) {
	_ = ctx

	operationID := node.OperationID()

	if operationID == "" {
		operationID = ir.OpID(node.OpType())
	}

	if operationID == ir.OpID(ir.OpInput) || node.OpType() == ir.OpInput {
		val, err := runner.materializeExternalInput(node, values)
		return val, nil, err
	}

	value, temps, deviceErr := runner.evaluateNodeDevice(node, values, weightsPath)

	if deviceErr == nil {
		return value, temps, nil
	}

	if !errors.Is(deviceErr, errDeviceDispatchUnsupported) {
		return nil, nil, fmt.Errorf("metal graph runner: node %q: %w", node.ID(), deviceErr)
	}

	kernelName := KernelName(operationID)

	kernel, ok := kernels.Default.LookupLocation(
		kernelName,
		metalKernelSignature(node),
		tensor.Metal,
	)

	if !ok {
		return nil, nil, fmt.Errorf("metal graph runner: no kernel for op %q (%s)", operationID, kernelName)
	}

	args, temps, err := runner.buildKernelArgs(node, values, weightsPath, kernel.Signature)

	if err != nil {
		return nil, nil, fmt.Errorf("metal graph runner: node %q build args: %w", node.ID(), err)
	}

	if err := kernel.Run(args...); err != nil {
		return nil, nil, fmt.Errorf("metal graph runner: node %q kernel %q: %w", node.ID(), kernelName, err)
	}

	outTensor := args[len(args)-len(kernel.Signature.Outputs):][0]

	return outTensor, temps, nil
}

func (runner *MetalGraphRunner) materializeExternalInput(
	node *ir.Node,
	values map[string]tensor.Tensor,
) (tensor.Tensor, error) {
	if value, ok := values[node.ID()]; ok {
		return value, nil
	}

	rawValues, ok := node.Metadata()["values"]

	if !ok {
		return nil, fmt.Errorf("metal graph runner: input node %s has no values", node.ID())
	}

	switch typed := rawValues.(type) {
	case []float64:
		return runner.memory.Upload(node.Shape(), dtype.Float32, float64ToFloat32Bytes(typed))
	case []int:
		buffer := make([]byte, len(typed)*4)

		for index, element := range typed {
			buffer[index*4] = byte(element)
			buffer[index*4+1] = byte(element >> 8)
			buffer[index*4+2] = byte(element >> 16)
			buffer[index*4+3] = byte(element >> 24)
		}

		return runner.memory.Upload(node.Shape(), dtype.Int32, buffer)
	case []int32:
		buffer := make([]byte, len(typed)*4)

		for index, element := range typed {
			value := uint32(element)
			buffer[index*4] = byte(value)
			buffer[index*4+1] = byte(value >> 8)
			buffer[index*4+2] = byte(value >> 16)
			buffer[index*4+3] = byte(value >> 24)
		}

		return runner.memory.Upload(node.Shape(), dtype.Int32, buffer)
	default:
		return nil, fmt.Errorf("metal graph runner: unsupported input values for %s", node.ID())
	}
}

func (runner *MetalGraphRunner) buildKernelArgs(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
	signature kernels.Signature,
) ([]tensor.Tensor, []tensor.Tensor, error) {
	operationID := node.OperationID()
	if operationID == "" {
		operationID = ir.OpID(node.OpType())
	}

	args := make([]tensor.Tensor, 0, len(signature.Inputs)+len(signature.Outputs))
	var temps []tensor.Tensor

	// Collect inputs from node.Inputs()
	nodeInputs := make([]tensor.Tensor, len(node.Inputs()))
	for i, inputNode := range node.Inputs() {
		value, ok := values[inputNode.ID()]
		if !ok {
			return nil, nil, fmt.Errorf("metal graph runner: missing input %q for %q", inputNode.ID(), node.ID())
		}
		nodeInputs[i] = value
	}

	// Collect weight if present
	var weightTensor tensor.Tensor
	weightName, hasWeightName := node.Metadata()["weight_name"].(string)

	if hasWeightName && weightsPath != "" {
		wt, err := runner.loadNodeWeight(node, weightsPath)

		if err != nil {
			return nil, nil, err
		}

		weightTensor = wt
	}

	if operationID == "embedding.token" && weightTensor == nil {
		return nil, nil, fmt.Errorf("embedding.token requires a weight tensor (weightName=%q, hasWeightName=%v, weightsPath=%q)", weightName, hasWeightName, weightsPath)
	}

	// Map to kernel arguments based on operation
	switch operationID {
	case "embedding.token":
		// [table, indices]
		if weightTensor == nil {
			return nil, nil, fmt.Errorf("embedding.token requires a weight tensor")
		}

		args = append(args, weightTensor, nodeInputs[0])

	case "projection.linear":
		// [input, weight, bias]
		if weightTensor == nil {
			return nil, nil, fmt.Errorf("projection.linear requires a weight tensor")
		}
		args = append(args, nodeInputs[0], weightTensor)
		biasTensor, err := runner.loadOptionalNodeBias(node, nodeInputs[0], weightsPath)
		if err != nil {
			return nil, nil, err
		}

		if biasTensor == nil {
			outFeatures := weightTensor.Shape().Dims()[0]
			biasShape, _ := tensor.NewShape([]int{outFeatures})
			biasTensor, err = runner.memory.NewZeroed(biasShape, weightTensor.DType())

			if err != nil {
				return nil, nil, err
			}

			temps = append(temps, biasTensor)
		}

		args = append(args, biasTensor)

	case "convolution.conv2d", "convolution.conv_transpose2d":
		if weightTensor == nil {
			return nil, nil, fmt.Errorf("%s requires a weight tensor", operationID)
		}

		args = append(args, nodeInputs[0], weightTensor)
		biasTensor, err := runner.loadOptionalNodeBias(node, nodeInputs[0], weightsPath)

		if err != nil {
			return nil, nil, err
		}

		if biasTensor == nil {
			outFeatures := weightTensor.Shape().Dims()[0]
			biasShape, _ := tensor.NewShape([]int{outFeatures})
			biasTensor, err = runner.memory.NewZeroed(biasShape, weightTensor.DType())

			if err != nil {
				return nil, nil, err
			}

			temps = append(temps, biasTensor)
		}

		args = append(args, biasTensor)

	case "math.rmsnorm", "math.layernorm":
		// [input, weight]
		if weightTensor == nil {
			return nil, nil, fmt.Errorf("%s requires a weight tensor", operationID)
		}
		args = append(args, nodeInputs[0], weightTensor)

	case "shape.view_as_heads":
		numHeadsAttr := node.Attribute("num_heads")
		if numHeadsAttr.Value == "" {
			return nil, nil, fmt.Errorf("shape.view_as_heads requires num_heads config")
		}

		var numHeads int32
		if parsed, err := strconv.ParseInt(numHeadsAttr.Value, 10, 32); err == nil {
			numHeads = int32(parsed)
		} else {
			return nil, nil, fmt.Errorf("shape.view_as_heads num_heads invalid: %v", err)
		}

		numHeadsShape, _ := tensor.NewShape([]int{1})
		numHeadsTensor, err := runner.memory.Upload(numHeadsShape, dtype.Int32, int32ToBytes([]int32{numHeads}))
		if err != nil {
			return nil, nil, err
		}
		temps = append(temps, numHeadsTensor)
		args = append(args, nodeInputs[0], numHeadsTensor)

	case "shape.slice":
		dimTensor, err := runner.int32Argument(node, "dim")

		if err != nil {
			return nil, nil, err
		}

		startTensor, err := runner.int32Argument(node, "start")

		if err != nil {
			return nil, nil, err
		}

		endTensor, err := runner.int32Argument(node, "end")

		if err != nil {
			return nil, nil, err
		}

		temps = append(temps, dimTensor, startTensor, endTensor)
		args = append(args, nodeInputs[0], dimTensor, startTensor, endTensor)

	case "shape.transpose":
		permutationTensor, err := runner.transposePermutation(node, nodeInputs[0].Shape())

		if err != nil {
			return nil, nil, err
		}

		temps = append(temps, permutationTensor)
		args = append(args, nodeInputs[0], permutationTensor)

	default:
		// Default: append all node inputs, then weight if present
		args = append(args, nodeInputs...)
		if weightTensor != nil {
			args = append(args, weightTensor)
		}
	}

	// Append output tensor
	outputDType := node.ValueType().DType
	if outputDType == dtype.Invalid {
		outputDType = dtype.Float32
	}

	inputShapes := make([]tensor.Shape, 0, len(nodeInputs))

	for _, nodeInput := range nodeInputs {
		inputShapes = append(inputShapes, nodeInput.Shape())
	}

	if weightTensor != nil {
		inputShapes = append(inputShapes, weightTensor.Shape())
	}

	outputShape, err := nodeOutputShape(node, inputShapes)

	if err != nil {
		return nil, nil, err
	}

	output, err := runner.memory.NewZeroed(outputShape, outputDType)
	if err != nil {
		return nil, nil, err
	}

	args = append(args, output)

	return args, temps, nil
}

func (runner *MetalGraphRunner) transposePermutation(
	node *ir.Node,
	inputShape tensor.Shape,
) (tensor.Tensor, error) {
	rank := len(inputShape.Dims())
	dim0 := int(int64Attribute(node, "dim0"))
	dim1 := int(int64Attribute(node, "dim1"))

	if rank == 0 || dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank {
		return nil, tensor.ErrShapeMismatch
	}

	permutation := make([]int32, rank)

	for dimensionIndex := range rank {
		permutation[dimensionIndex] = int32(dimensionIndex)
	}

	permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
	shape, _ := tensor.NewShape([]int{rank})

	return runner.memory.Upload(shape, dtype.Int32, int32ToBytes(permutation))
}

func (runner *MetalGraphRunner) int32Argument(
	node *ir.Node,
	name string,
) (tensor.Tensor, error) {
	value, err := strconv.ParseInt(node.Attribute(name).Value, 10, 32)

	if err != nil {
		return nil, fmt.Errorf("%s requires %s config: %w", node.OperationID(), name, err)
	}

	argumentShape, _ := tensor.NewShape([]int{1})

	return runner.memory.Upload(argumentShape, dtype.Int32, int32ToBytes([]int32{int32(value)}))
}

func weightPathForNode(node *ir.Node, graphWeightsPath string) string {
	if weightFile, ok := node.Metadata()["weight_file"].(string); ok && weightFile != "" {
		return weightFile
	}

	return graphWeightsPath
}

func (runner *MetalGraphRunner) loadWeightTensor(
	weightsPath string,
	weightName string,
	format dtype.DType,
	fallbackShape tensor.Shape,
) (tensor.Tensor, error) {
	if cached, ok := runner.weightsCache[weightName]; ok {
		return cached, nil
	}

	st, err := hub.OpenSafeTensors(weightsPath)
	if err != nil {
		return nil, err
	}
	defer st.Close()

	raw, meta, err := st.Tensor(weightName)
	if err != nil {
		return nil, err
	}

	weightDType, err := dtype.Parse(meta.DType)

	if err != nil {
		return nil, err
	}

	shapeDims := make([]int, len(meta.Shape))

	for index, dimension := range meta.Shape {
		shapeDims[index] = int(dimension)
	}

	shape, err := tensor.NewShape(shapeDims)

	if err != nil {
		shape = fallbackShape
	}

	if format == dtype.Invalid {
		format = weightDType
	}

	storageDType := weightDType

	if format != weightDType && format == dtype.Float32 {
		float32s, convertErr := convert.BytesToFloat32(weightDType, raw)

		if convertErr != nil {
			return nil, convertErr
		}

		raw = convert.Float32ToBytes(float32s)
		storageDType = dtype.Float32
	}

	tensorValue, err := runner.memory.Upload(shape, storageDType, raw)
	if err == nil {
		runner.weightsCache[weightName] = tensorValue
	}
	return tensorValue, err
}

func (runner *MetalGraphRunner) loadWeightTensorSlice(
	weightsPath string,
	weightName string,
	format dtype.DType,
	sliceAxis string,
	start int64,
	end int64,
) (tensor.Tensor, error) {
	cacheKey := fmt.Sprintf("%s[%s:%d:%d]", weightName, sliceAxis, start, end)

	if cached, ok := runner.weightsCache[cacheKey]; ok {
		return cached, nil
	}

	st, err := hub.OpenSafeTensors(weightsPath)

	if err != nil {
		return nil, err
	}

	defer st.Close()

	raw, meta, err := st.Tensor(weightName)

	if err != nil {
		return nil, err
	}

	weightDType, err := dtype.Parse(meta.DType)

	if err != nil {
		return nil, err
	}

	size, err := weightDType.Size()

	if err != nil {
		return nil, err
	}

	raw, shapeDims, err := slicedWeightBytes(raw, meta.Shape, size, sliceAxis, start, end)

	if err != nil {
		return nil, err
	}

	shape, err := tensor.NewShape(shapeDims)

	if err != nil {
		return nil, err
	}

	storageDType := weightDType

	if format != weightDType && format == dtype.Float32 {
		float32s, convertErr := convert.BytesToFloat32(weightDType, raw)

		if convertErr != nil {
			return nil, convertErr
		}

		raw = convert.Float32ToBytes(float32s)
		storageDType = dtype.Float32
	}

	tensorValue, err := runner.memory.Upload(shape, storageDType, raw)

	if err == nil {
		runner.weightsCache[cacheKey] = tensorValue
	}

	return tensorValue, err
}

func slicedWeightBytes(
	raw []byte,
	metaShape []int64,
	elementSize int,
	sliceAxis string,
	start int64,
	end int64,
) ([]byte, []int, error) {
	if len(metaShape) != 2 || start < 0 || end <= start {
		return nil, nil, tensor.ErrShapeMismatch
	}

	rows := int(metaShape[0])
	columns := int(metaShape[1])
	startIndex := int(start)
	endIndex := int(end)

	switch sliceAxis {
	case "output":
		if endIndex > rows {
			return nil, nil, tensor.ErrShapeMismatch
		}

		rowBytes := columns * elementSize
		begin := startIndex * rowBytes
		finish := endIndex * rowBytes

		return raw[begin:finish], []int{endIndex - startIndex, columns}, nil
	case "input":
		if endIndex > columns {
			return nil, nil, tensor.ErrShapeMismatch
		}

		out := make([]byte, rows*(endIndex-startIndex)*elementSize)
		sourceRowBytes := columns * elementSize
		targetRowBytes := (endIndex - startIndex) * elementSize

		for rowIndex := range rows {
			sourceStart := rowIndex*sourceRowBytes + startIndex*elementSize
			sourceEnd := sourceStart + targetRowBytes
			targetStart := rowIndex * targetRowBytes
			copy(out[targetStart:targetStart+targetRowBytes], raw[sourceStart:sourceEnd])
		}

		return out, []int{rows, endIndex - startIndex}, nil
	default:
		return nil, nil, tensor.ErrShapeMismatch
	}
}

func metalKernelSignature(node *ir.Node) kernels.Signature {
	operationID := node.OperationID()
	if operationID == "" {
		operationID = ir.OpID(node.OpType())
	}

	valueType := node.ValueType()
	format := valueType.DType
	if format == dtype.Invalid {
		format = dtype.Float32
	}

	if operationID == "embedding.token" {
		return kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{format, dtype.Int32},
			Outputs: []dtype.DType{format},
		}
	}

	if operationID == "shape.view_as_heads" {
		return kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{format, dtype.Int32},
			Outputs: []dtype.DType{format},
		}
	}

	if operationID == "shape.slice" {
		return kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				format,
				dtype.Int32,
				dtype.Int32,
				dtype.Int32,
			},
			Outputs: []dtype.DType{format},
		}
	}

	if operationID == "shape.transpose" {
		return kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{format, dtype.Int32},
			Outputs: []dtype.DType{format},
		}
	}

	if operationID == "projection.linear" ||
		operationID == "convolution.conv2d" ||
		operationID == "convolution.conv_transpose2d" {
		return kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{format, format, format},
			Outputs: []dtype.DType{format},
		}
	}

	inputs := make([]dtype.DType, len(node.Inputs()))
	for index := range node.Inputs() {
		inputs[index] = format
	}

	if _, ok := node.Metadata()["weight_name"]; ok {
		inputs = append(inputs, format)
	}

	return kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  inputs,
		Outputs: []dtype.DType{format},
	}
}
func float64ToFloat32Bytes(values []float64) []byte {
	buffer := make([]byte, len(values)*4)

	for index, element := range values {
		bits := math.Float32bits(float32(element))
		buffer[index*4] = byte(bits)
		buffer[index*4+1] = byte(bits >> 8)
		buffer[index*4+2] = byte(bits >> 16)
		buffer[index*4+3] = byte(bits >> 24)
	}

	return buffer
}

func int32ToBytes(values []int32) []byte {
	buffer := make([]byte, len(values)*4)
	for index, element := range values {
		value := uint32(element)
		buffer[index*4] = byte(value)
		buffer[index*4+1] = byte(value >> 8)
		buffer[index*4+2] = byte(value >> 16)
		buffer[index*4+3] = byte(value >> 24)
	}
	return buffer
}
