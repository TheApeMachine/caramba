package runtime

import (
	"context"
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/runtime/weights"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

/*
GraphRunner executes lowered IR graphs on the host using puter CPU kernels.
*/
type GraphRunner struct {
	memory tensor.Backend
}

/*
NewGraphRunner constructs a GraphRunner backed by host tensor memory.
*/
func NewGraphRunner(memory tensor.Backend) *GraphRunner {
	return &GraphRunner{memory: memory}
}

/*
Execute evaluates all graph nodes and returns requested target outputs.
*/
func (runner *GraphRunner) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
	weightsPath string,
	externalInputs map[string]tensor.Tensor,
) (map[string]tensor.Tensor, error) {
	if graph == nil {
		return nil, fmt.Errorf("graph runner: graph is required")
	}

	layers, err := graph.TopologyLayers()

	if err != nil {
		return nil, err
	}

	values := make(map[string]tensor.Tensor, len(graph.Nodes()))

	for key, value := range externalInputs {
		values[key] = value
	}

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

			return nil, fmt.Errorf("graph runner: target %q produced no output", target.ID())
		}

		outputs[target.ID()] = value
	}

	return outputs, nil
}

func (runner *GraphRunner) evaluateNode(
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
		kernelSignature(node),
		tensor.Host,
	)

	if !ok {
		return nil, fmt.Errorf("graph runner: no kernel for op %q (%s)", operationID, kernelName)
	}

	args, err := runner.buildKernelArgs(node, values, weightsPath, kernel.Signature)

	if err != nil {
		return nil, err
	}

	if err := kernel.Run(args...); err != nil {
		return nil, fmt.Errorf("graph runner: kernel %q: %w", kernelName, err)
	}

	return args[len(args)-len(kernel.Signature.Outputs):][0], nil
}

func (runner *GraphRunner) materializeExternalInput(
	node *ir.Node,
	values map[string]tensor.Tensor,
) (tensor.Tensor, error) {
	if value, ok := values[node.ID()]; ok {
		return value, nil
	}

	rawValues, ok := node.Metadata()["values"]

	if !ok {
		return nil, fmt.Errorf("graph runner: input node %s has no values", node.ID())
	}

	switch typed := rawValues.(type) {
	case []float64:
		bytes := make([]byte, len(typed)*8)

		for index, element := range typed {
			bits := math.Float64bits(element)
			offset := index * 8
			bytes[offset] = byte(bits)
			bytes[offset+1] = byte(bits >> 8)
			bytes[offset+2] = byte(bits >> 16)
			bytes[offset+3] = byte(bits >> 24)
			bytes[offset+4] = byte(bits >> 32)
			bytes[offset+5] = byte(bits >> 40)
			bytes[offset+6] = byte(bits >> 48)
			bytes[offset+7] = byte(bits >> 56)
		}

		return runner.memory.Upload(node.Shape(), dtype.Float64, bytes)
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
		return nil, fmt.Errorf("graph runner: unsupported input values for %s", node.ID())
	}
}

func (runner *GraphRunner) buildKernelArgs(
	node *ir.Node,
	values map[string]tensor.Tensor,
	weightsPath string,
	signature kernels.Signature,
) ([]tensor.Tensor, error) {
	inputCount := len(signature.Inputs)
	outputCount := len(signature.Outputs)
	args := make([]tensor.Tensor, 0, inputCount+outputCount)

	for _, inputNode := range node.Inputs() {
		value, ok := values[inputNode.ID()]

		if !ok {
			return nil, fmt.Errorf("graph runner: missing input %q for %q", inputNode.ID(), node.ID())
		}

		args = append(args, value)
	}

	if weightName, ok := node.Metadata()["weight_name"].(string); ok && weightsPath != "" {
		weightTensor, err := runner.loadWeightTensor(weightsPath, weightName, node.ValueType().DType, node.Shape())

		if err != nil {
			return nil, err
		}

		args = append(args, weightTensor)
	}

	outputShape := node.Shape()
	outputDType := node.ValueType().DType

	if outputDType == dtype.Invalid {
		outputDType = dtype.Float32
	}

	output, err := tensor.NewZeroed(outputShape, outputDType)

	if err != nil {
		return nil, err
	}

	args = append(args, output)

	return args, nil
}

func (runner *GraphRunner) loadWeightTensor(
	weightsPath string,
	weightName string,
	format dtype.DType,
	fallbackShape tensor.Shape,
) (tensor.Tensor, error) {
	raw, meta, err := weights.ReadTensor(weightsPath, weightName)

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

	return runner.memory.Upload(shape, weightDType, raw)
}

func kernelSignature(node *ir.Node) kernels.Signature {
	valueType := node.ValueType()
	inputs := make([]dtype.DType, len(node.Inputs())+1)
	outputs := []dtype.DType{dtype.Float32}

	format := valueType.DType

	if format == dtype.Invalid {
		format = dtype.Float32
	}

	for index := range node.Inputs() {
		inputs[index] = format
	}

	if _, ok := node.Metadata()["weight_name"]; ok {
		inputs[len(node.Inputs())] = format
	}

	return kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  inputs,
		Outputs: outputs,
	}
}
