package executor

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Backend is the resident tensor kernel surface required by graph execution.
Every network-shared compute backend must satisfy this interface without
falling back to a different location.
*/
type Backend interface {
	tensor.Float64ActivationBackend
	tensor.Float64MathBackend
	tensor.Float64FusedBackend
}

type TensorSpec struct {
	ID    string
	Shape []int64
	Data  []byte
	DType tensor.DType
}

type NodeSpec struct {
	ID       string
	Op       ir.OpType
	Inputs   []string
	Shape    []int
	Metadata map[string]any
	Target   bool
}

type Executor struct {
	backend  Backend
	registry *Registry
	values   map[string]tensor.Float64Tensor
	nodes    map[string]NodeSpec
	states   map[string]int
}

func New(backend Backend) *Executor {
	return NewWithRegistry(backend, NewTensorRegistry())
}

func NewWithRegistry(backend Backend, registry *Registry) *Executor {
	if registry == nil {
		registry = NewTensorRegistry()
	}

	return &Executor{
		backend:  backend,
		registry: registry,
		values:   make(map[string]tensor.Float64Tensor),
		nodes:    make(map[string]NodeSpec),
		states:   make(map[string]int),
	}
}

func (executor *Executor) Execute(
	ctx context.Context, nodes []NodeSpec, tensors []TensorSpec,
) ([]TensorSpec, error) {
	if executor.backend == nil {
		return nil, fmt.Errorf("executor: backend is required")
	}

	for _, node := range nodes {
		executor.nodes[node.ID] = node
	}

	for _, tensorSpec := range tensors {
		uploaded, err := executor.upload(tensorSpec)

		if err != nil {
			return nil, err
		}

		executor.values[tensorSpec.ID] = uploaded
	}

	targets := targets(nodes)
	outputs := make([]TensorSpec, len(targets))

	for index, target := range targets {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		value, err := executor.evaluate(ctx, target.ID)

		if err != nil {
			return nil, err
		}

		output, err := executor.download(target.ID, value)

		if err != nil {
			return nil, err
		}

		outputs[index] = output
	}

	return outputs, nil
}

func (executor *Executor) upload(tensorSpec TensorSpec) (tensor.Float64Tensor, error) {
	if tensorSpec.DType != tensor.Float64 {
		return nil, fmt.Errorf("executor: unsupported tensor dtype %q", tensorSpec.DType)
	}

	values, err := DecodeFloat64(tensorSpec.Data)

	if err != nil {
		return nil, err
	}

	shape, err := shapeFromInt64(tensorSpec.Shape)

	if err != nil {
		return nil, err
	}

	return executor.backend.UploadFloat64(shape, values)
}

func (executor *Executor) evaluate(ctx context.Context, id string) (tensor.Float64Tensor, error) {
	if value, ok := executor.values[id]; ok {
		return value, nil
	}

	switch executor.states[id] {
	case 1:
		return nil, fmt.Errorf("executor: cycle detected at node %q", id)
	case 2:
		return executor.values[id], nil
	}

	node, ok := executor.nodes[id]

	if !ok {
		return nil, fmt.Errorf("executor: unknown node %q", id)
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	executor.states[id] = 1
	inputs := make([]tensor.Float64Tensor, len(node.Inputs))

	for index, inputID := range node.Inputs {
		input, err := executor.evaluate(ctx, inputID)

		if err != nil {
			return nil, err
		}

		inputs[index] = input
	}

	output, err := executor.apply(ctx, node, inputs)

	if err != nil {
		return nil, err
	}

	executor.values[id] = output
	executor.states[id] = 2

	return output, nil
}

func (executor *Executor) apply(
	ctx context.Context, node NodeSpec, inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	handler, ok := executor.registry.Handler(node.Op)
	if !ok {
		return nil, fmt.Errorf("executor: unsupported operation %q", node.Op)
	}

	return handler(ctx, executor.backend, node, inputs)
}

func (executor *Executor) download(
	id string, value tensor.Float64Tensor,
) (TensorSpec, error) {
	values, err := value.CloneFloat64()

	if err != nil {
		return TensorSpec{}, err
	}

	data, err := EncodeFloat64(values)

	if err != nil {
		return TensorSpec{}, err
	}

	dimensions := value.Shape().Dims()
	shape := make([]int64, len(dimensions))

	for index, dimension := range dimensions {
		shape[index] = int64(dimension)
	}

	return TensorSpec{
		ID:    id,
		Shape: shape,
		Data:  data,
		DType: tensor.Float64,
	}, nil
}

func targets(nodes []NodeSpec) []NodeSpec {
	targetNodes := make([]NodeSpec, 0)
	consumed := make(map[string]bool)

	for _, node := range nodes {
		if node.Target {
			targetNodes = append(targetNodes, node)
		}

		for _, inputID := range node.Inputs {
			consumed[inputID] = true
		}
	}

	if len(targetNodes) > 0 {
		return targetNodes
	}

	for _, node := range nodes {
		if !consumed[node.ID] {
			targetNodes = append(targetNodes, node)
		}
	}

	return targetNodes
}

func EncodeFloat64(values []float64) ([]byte, error) {
	buffer := bytes.NewBuffer(make([]byte, 0, len(values)*8))

	for _, value := range values {
		if err := binary.Write(buffer, binary.LittleEndian, value); err != nil {
			return nil, err
		}
	}

	return buffer.Bytes(), nil
}

func DecodeFloat64(data []byte) ([]float64, error) {
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("executor: float64 tensor data must be divisible by 8")
	}

	values := make([]float64, len(data)/8)
	reader := bytes.NewReader(data)

	for index := range values {
		if err := binary.Read(reader, binary.LittleEndian, &values[index]); err != nil {
			return nil, err
		}
	}

	return values, nil
}

func shapeFromInt64(shape []int64) (tensor.Shape, error) {
	dimensions := make([]int, len(shape))
	maxDimension := int64(int(^uint(0) >> 1))

	for index, dimension := range shape {
		if dimension > maxDimension {
			return tensor.Shape{}, fmt.Errorf("executor: tensor dimension overflows int")
		}

		dimensions[index] = int(dimension)
	}

	return tensor.NewShape(dimensions)
}
