package executor

import (
	"context"
	"fmt"
	"strings"

	cpuoperation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type Handler func(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error)

type Registry struct {
	handlers map[ir.OpType]Handler
}

func NewRegistry() *Registry {
	return &Registry{
		handlers: make(map[ir.OpType]Handler),
	}
}

func NewTensorRegistry() *Registry {
	registry := NewRegistry()
	registry.Register(ir.OpInput, inputHandler)
	registry.Register(ir.OpAdd, exactInputCount(2, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.Add(inputs[0], inputs[1])
	}))
	registry.Register(ir.OpMul, exactInputCount(2, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.Mul(inputs[0], inputs[1])
	}))
	registry.Register(ir.OpMatmul, exactInputCount(2, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.Matmul(inputs[0], inputs[1])
	}))
	registry.Register(ir.OpReLU, exactInputCount(1, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.ReLU(inputs[0])
	}))
	registry.Register(ir.OpLeakyReLU, exactInputCount(1, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.LeakyReLU(inputs[0], 0.01)
	}))
	registry.Register(ir.OpGELU, exactInputCount(1, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.GELU(inputs[0])
	}))
	registry.Register(ir.OpTanh, exactInputCount(1, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.Tanh(inputs[0])
	}))
	registry.Register(ir.OpSigmoid, exactInputCount(1, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.Sigmoid(inputs[0])
	}))
	registry.Register(ir.OpSwiGLU, exactInputCount(1, func(
		backend Backend, inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		return backend.SwiGLU(inputs[0])
	}))
	registry.Register(ir.OpFused, fusedHandler)

	return registry
}

func (registry *Registry) Register(op ir.OpType, handler Handler) {
	if registry == nil || handler == nil {
		return
	}

	registry.handlers[normalizeOp(op)] = handler
}

func (registry *Registry) Merge(other *Registry) {
	if registry == nil || other == nil {
		return
	}

	for op, handler := range other.handlers {
		registry.handlers[op] = handler
	}
}

func (registry *Registry) Handler(op ir.OpType) (Handler, bool) {
	if registry == nil {
		return nil, false
	}

	handler, ok := registry.handlers[normalizeOp(op)]

	return handler, ok
}

func (registry *Registry) Supports(op ir.OpType) bool {
	_, ok := registry.Handler(op)

	return ok
}

func OperationHandler(operation cpuoperation.Operation) Handler {
	return func(
		ctx context.Context,
		backend Backend,
		node NodeSpec,
		inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		values, err := inputValues(inputs)
		if err != nil {
			return nil, err
		}

		output := operation.Forward(outputShape(node, inputs), values...)
		shape, err := outputTensorShape(node, inputs, output)

		if err != nil {
			return nil, err
		}

		return backend.UploadFloat64(shape, output)
	}
}

func ErrorOperationHandler(
	operation func(shape []int, data ...[]float64) ([]float64, error),
) Handler {
	return func(
		ctx context.Context,
		backend Backend,
		node NodeSpec,
		inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		values, err := inputValues(inputs)
		if err != nil {
			return nil, err
		}

		output, err := operation(outputShape(node, inputs), values...)
		if err != nil {
			return nil, err
		}

		shape, err := outputTensorShape(node, inputs, output)
		if err != nil {
			return nil, err
		}

		return backend.UploadFloat64(shape, output)
	}
}

func inputHandler(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	_ = backend
	_ = inputs

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, fmt.Errorf("executor: input node %q was not materialized", node.ID)
}

func exactInputCount(
	count int,
	apply func(Backend, []tensor.Float64Tensor) (tensor.Float64Tensor, error),
) Handler {
	return func(
		ctx context.Context,
		backend Backend,
		node NodeSpec,
		inputs []tensor.Float64Tensor,
	) (tensor.Float64Tensor, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if len(inputs) != count {
			return nil, fmt.Errorf(
				"executor: %s node %q requires %d inputs",
				node.Op,
				node.ID,
				count,
			)
		}

		return apply(backend, inputs)
	}
}

func fusedHandler(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("executor: Fused node %q requires 3 inputs", node.ID)
	}

	activation, _ := node.Metadata["activation"].(string)
	if strings.EqualFold(activation, string(ir.OpGELU)) {
		return backend.MatmulAddGELU(inputs[0], inputs[1], inputs[2])
	}

	return backend.MatmulAdd(inputs[0], inputs[1], inputs[2])
}

func inputValues(inputs []tensor.Float64Tensor) ([][]float64, error) {
	values := make([][]float64, len(inputs))

	for index, input := range inputs {
		value, err := input.CloneFloat64()

		if err != nil {
			return nil, err
		}

		values[index] = value
	}

	return values, nil
}

func outputShape(node NodeSpec, inputs []tensor.Float64Tensor) []int {
	if len(node.Shape) > 0 {
		return append([]int(nil), node.Shape...)
	}

	if len(inputs) == 0 {
		return nil
	}

	return inputs[0].Shape().Dims()
}

func outputTensorShape(
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	output []float64,
) (tensor.Shape, error) {
	shapeData := outputShape(node, inputs)
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

func normalizeOp(op ir.OpType) ir.OpType {
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
	case "fused", "math.matmul_add", "math.matmul_add_gelu":
		return ir.OpFused
	default:
		return op
	}
}
