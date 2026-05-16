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
	tensor.Backend
	Apply(
		ctx context.Context,
		node NodeSpec,
		inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error)
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
	backend   Backend
	values    map[string]tensor.Float64Tensor
	nodes     map[string]NodeSpec
	states    map[string]int
	owned     map[string]bool
	remaining map[string]int
	keep      map[string]bool
}

func New(backend Backend) *Executor {
	return &Executor{
		backend:   backend,
		values:    make(map[string]tensor.Float64Tensor),
		nodes:     make(map[string]NodeSpec),
		states:    make(map[string]int),
		owned:     make(map[string]bool),
		remaining: make(map[string]int),
		keep:      make(map[string]bool),
	}
}

func (executor *Executor) Execute(
	ctx context.Context, nodes []NodeSpec, tensors []TensorSpec,
) (map[string]tensor.Float64Tensor, error) {
	if err := executor.reset(); err != nil {
		return nil, err
	}

	if executor.backend == nil {
		return nil, fmt.Errorf("executor: backend is required")
	}

	var callerOwnedOutputs map[string]tensor.Float64Tensor
	defer func() {
		if callerOwnedOutputs == nil {
			_ = executor.closeOwnedExcept(nil)
		}
	}()

	for _, node := range nodes {
		executor.nodes[node.ID] = node
	}

	targetSpecs := targets(nodes)
	executor.prepareLifetime(nodes, targetSpecs)

	for _, tensorSpec := range tensors {
		uploaded, err := executor.upload(tensorSpec)

		if err != nil {
			return nil, err
		}

		executor.values[tensorSpec.ID] = uploaded
		executor.owned[tensorSpec.ID] = true
	}

	outputs := make(map[string]tensor.Float64Tensor, len(targetSpecs))
	outputIDs := make(map[string]bool, len(targetSpecs))

	for _, target := range targetSpecs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		value, err := executor.evaluate(ctx, target.ID)

		if err != nil {
			return nil, err
		}

		outputs[target.ID] = value
		outputIDs[target.ID] = true
	}

	if err := executor.closeOwnedExcept(outputIDs); err != nil {
		return nil, err
	}

	executor.detach(outputIDs)
	callerOwnedOutputs = outputs

	return outputs, nil
}

func (executor *Executor) reset() error {
	if len(executor.values) != 0 {
		if err := executor.closeOwnedExcept(nil); err != nil {
			return err
		}
	}

	executor.values = make(map[string]tensor.Float64Tensor)
	executor.nodes = make(map[string]NodeSpec)
	executor.states = make(map[string]int)
	executor.owned = make(map[string]bool)
	executor.remaining = make(map[string]int)
	executor.keep = make(map[string]bool)

	return nil
}

func (executor *Executor) prepareLifetime(nodes []NodeSpec, targets []NodeSpec) {
	executor.remaining = make(map[string]int)
	executor.keep = make(map[string]bool, len(targets))
	reachable := make(map[string]bool, len(nodes))

	for _, target := range targets {
		executor.markReachable(target.ID, reachable)
	}

	for _, node := range nodes {
		if !reachable[node.ID] {
			continue
		}

		for _, inputID := range node.Inputs {
			executor.remaining[inputID]++
		}
	}

	for _, target := range targets {
		executor.keep[target.ID] = true
	}
}

func (executor *Executor) markReachable(id string, reachable map[string]bool) {
	if reachable[id] {
		return
	}

	reachable[id] = true
	node, ok := executor.nodes[id]

	if !ok {
		return
	}

	for _, inputID := range node.Inputs {
		executor.markReachable(inputID, reachable)
	}
}

func (executor *Executor) closeOwnedExcept(keep map[string]bool) error {
	var closeErr error

	for id, value := range executor.values {
		if keep[id] {
			continue
		}

		if !executor.owned[id] || value == nil {
			continue
		}

		if err := value.Close(); err != nil && closeErr == nil {
			closeErr = err
		}

		delete(executor.owned, id)
		delete(executor.values, id)
	}

	return closeErr
}

func (executor *Executor) detach(ids map[string]bool) {
	for id := range ids {
		delete(executor.owned, id)
		delete(executor.values, id)
	}
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
		value := executor.values[id]

		if value == nil {
			return nil, fmt.Errorf("executor: node %q was released before evaluation completed", id)
		}

		return value, nil
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
	executor.owned[id] = true
	executor.states[id] = 2

	for _, inputID := range node.Inputs {
		if err := executor.releaseAfterUse(inputID); err != nil {
			return nil, err
		}
	}

	return output, nil
}

func (executor *Executor) releaseAfterUse(id string) error {
	remaining, ok := executor.remaining[id]

	if !ok {
		return nil
	}

	if remaining <= 0 {
		return fmt.Errorf("executor: node %q lifetime underflow", id)
	}

	remaining--
	executor.remaining[id] = remaining

	if remaining != 0 || executor.keep[id] || !executor.owned[id] {
		return nil
	}

	value := executor.values[id]

	if value == nil {
		return fmt.Errorf("executor: owned node %q has no resident value", id)
	}

	if err := value.Close(); err != nil {
		return err
	}

	delete(executor.owned, id)
	delete(executor.values, id)

	return nil
}

func (executor *Executor) apply(
	ctx context.Context, node NodeSpec, inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	node = WithDerivedMetadata(node, inputs)

	return executor.backend.Apply(ctx, node, inputs)
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
