package runtime

import (
	"context"
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
	"github.com/theapemachine/puter/kernels"
)

/*
MetalGraphRunner executes lowered IR graphs on Metal through direct kernel dispatch.
*/
type MetalGraphRunner struct {
	memory *metal.Backend
}

/*
NewMetalGraphRunner constructs a MetalGraphRunner.
*/
func NewMetalGraphRunner(memory *metal.Backend) *MetalGraphRunner {
	return &MetalGraphRunner{memory: memory}
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
	defer runner.memory.EndBatch()

	for _, layer := range layers {
		for _, node := range layer {
			if err := ctx.Err(); err != nil {
				closeValues(values)

				return nil, err
			}

			value, evalErr := runner.evaluateNode(ctx, node, values, weightsPath)

			if evalErr != nil {
				closeValues(values)

				return nil, evalErr
			}

			if value != nil {
				values[node.ID()] = value
			}
		}
	}

	outputs := make(map[string]tensor.Tensor, len(targets))

	for _, target := range targets {
		value, ok := values[target.ID()]

		if !ok || value == nil {
			closeValues(values)

			return nil, fmt.Errorf("metal graph runner: target %q produced no output", target.ID())
		}

		outputs[target.ID()] = value
	}

	return outputs, nil
}

func (runner *MetalGraphRunner) evaluateNode(
	ctx context.Context,
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
) (tensor.Tensor, error) {
	_ = ctx

	operationID := node.OperationID()

	if operationID == "" {
		operationID = ir.OpID(node.OpType())
	}

	if operationID == ir.OpID(ir.OpInput) || node.OpType() == ir.OpInput {
		return runner.materializeExternalInput(node, values)
	}

	kernelName := KernelName(operationID)

	kernel, ok := kernels.Default.LookupLocation(
		kernelName,
		metalKernelSignature(node),
		tensor.Metal,
	)

	if !ok {
		return nil, fmt.Errorf("metal graph runner: no kernel for op %q (%s)", operationID, kernelName)
	}

	args, err := runner.buildKernelArgs(node, values, weightsPath, kernel.Signature)

	if err != nil {
		return nil, err
	}

	if err := kernel.Run(args...); err != nil {
		return nil, fmt.Errorf("metal graph runner: kernel %q: %w", kernelName, err)
	}

	return args[len(args)-len(kernel.Signature.Outputs):][0], nil
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
) ([]tensor.Tensor, error) {
	operationID := node.OperationID()
	if operationID == "" {
		operationID = ir.OpID(node.OpType())
	}

	args := make([]tensor.Tensor, 0, len(signature.Inputs)+len(signature.Outputs))

	// Collect inputs from node.Inputs()
	nodeInputs := make([]tensor.Tensor, len(node.Inputs()))
	for i, inputNode := range node.Inputs() {
		value, ok := values[inputNode.ID()]
		if !ok {
			return nil, fmt.Errorf("metal graph runner: missing input %q for %q", inputNode.ID(), node.ID())
		}
		nodeInputs[i] = value
	}

	// Collect weight if present
	var weightTensor tensor.Tensor
	weightName, hasWeightName := node.Metadata()["weight_name"].(string)
	if hasWeightName && weightsPath != "" {
		wt, err := runner.loadWeightTensor(weightsPath, weightName, node.ValueType().DType, node.Shape())
		if err != nil {
			return nil, err
		}
		weightTensor = wt
	}

	if operationID == "embedding.token" && weightTensor == nil {
		return nil, fmt.Errorf("embedding.token requires a weight tensor (weightName=%q, hasWeightName=%v, weightsPath=%q)", weightName, hasWeightName, weightsPath)
	}

	// Map to kernel arguments based on operation
	switch operationID {
	case "embedding.token":
		// [table, indices]
		if weightTensor == nil {
			return nil, fmt.Errorf("embedding.token requires a weight tensor")
		}
		
		args = append(args, weightTensor, nodeInputs[0])

	case "projection.linear":
		// [input, weight, bias]
		if weightTensor == nil {
			return nil, fmt.Errorf("projection.linear requires a weight tensor")
		}
		args = append(args, nodeInputs[0], weightTensor)
		// Bias is optional in manifesto, but required by puter kernel.
		// Create a zero bias tensor of shape [out_features].
		outFeatures := weightTensor.Shape().Dims()[0]
		biasShape, _ := tensor.NewShape([]int{outFeatures})
		biasTensor, err := runner.memory.NewZeroed(biasShape, weightTensor.DType())
		if err != nil {
			return nil, err
		}
		args = append(args, biasTensor)

	case "math.rmsnorm", "math.layernorm":
		// [input, weight]
		if weightTensor == nil {
			return nil, fmt.Errorf("%s requires a weight tensor", operationID)
		}
		args = append(args, nodeInputs[0], weightTensor)

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

	outputShape := node.Shape()
	if len(nodeInputs) > 0 {
		seqLen := nodeInputs[0].Shape().Dims()[0]
		if operationID == "embedding.token" {
			seqLen = nodeInputs[0].Shape().Len()
		}
		
		dims := outputShape.Dims()
		if len(dims) > 0 && dims[0] != seqLen {
			newDims := make([]int, len(dims))
			copy(newDims, dims)
			newDims[0] = seqLen
			outputShape, _ = tensor.NewShape(newDims)
		}
	}

	output, err := runner.memory.NewZeroed(outputShape, outputDType)
	if err != nil {
		return nil, err
	}

	args = append(args, output)

	return args, nil
}

func (runner *MetalGraphRunner) loadWeightTensor(
	weightsPath string,
	weightName string,
	format dtype.DType,
	fallbackShape tensor.Shape,
) (tensor.Tensor, error) {
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

	if format == dtype.Float32 && weightDType != dtype.Float32 {
		float32s, err := convert.BytesToFloat32(weightDType, raw)
		if err != nil {
			return nil, err
		}
		raw = convert.Float32ToBytes(float32s)
		weightDType = dtype.Float32
	}

	return runner.memory.Upload(shape, weightDType, raw)
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

	if operationID == "projection.linear" {
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
