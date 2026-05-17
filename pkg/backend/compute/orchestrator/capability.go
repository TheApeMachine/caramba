package orchestrator

import (
	"slices"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type Cost struct {
	FLOPs        uint64
	BytesRead    uint64
	BytesWritten uint64
}

type Capabilities interface {
	Location() tensor.Location
	Supports(op ir.OpType) bool
	Precision(op ir.OpType) tensor.DType
	CanFuse(pattern string, fused ir.OpType) bool
	Cost(node *ir.Node) Cost
}

type CapabilityProvider interface {
	Capabilities() Capabilities
}

type StaticCapabilities struct {
	location  tensor.Location
	ops       map[ir.OpType]bool
	precision map[ir.OpType]tensor.DType
	fusions   map[string]map[ir.OpType]bool
	shapes    map[ir.OpType][]string
}

func NewStaticCapabilities(location tensor.Location) *StaticCapabilities {
	return &StaticCapabilities{
		location:  location,
		ops:       make(map[ir.OpType]bool),
		precision: make(map[ir.OpType]tensor.DType),
		fusions:   make(map[string]map[ir.OpType]bool),
		shapes:    make(map[ir.OpType][]string),
	}
}

func NewDefaultCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)
	capabilities.Register(
		ir.OpInput,
		ir.OpAdd,
		ir.OpMul,
		ir.OpMatmul,
		ir.OpReLU,
		ir.OpLeakyReLU,
		ir.OpGELU,
		ir.OpTanh,
		ir.OpSigmoid,
		ir.OpSwiGLU,
		ir.OpFused,
	)
	capabilities.RegisterFusion("matmul.activation", ir.OpFused)

	return capabilities
}

func (capabilities *StaticCapabilities) Location() tensor.Location {
	return capabilities.location
}

func (capabilities *StaticCapabilities) Register(ops ...ir.OpType) {
	for _, op := range ops {
		capabilities.ops[op] = true
	}
}

func (capabilities *StaticCapabilities) Supports(op ir.OpType) bool {
	return capabilities.ops["*"] || capabilities.ops[op]
}

func (capabilities *StaticCapabilities) Precision(op ir.OpType) tensor.DType {
	if precision := capabilities.precision[op]; precision != "" {
		return precision
	}

	if precision := capabilities.precision["*"]; precision != "" {
		return precision
	}

	return tensor.Float64
}

func (capabilities *StaticCapabilities) SetPrecision(precision tensor.DType, ops ...ir.OpType) {
	if len(ops) == 0 {
		capabilities.precision["*"] = precision

		return
	}

	for _, op := range ops {
		capabilities.precision[op] = precision
	}
}

func (capabilities *StaticCapabilities) RegisterFusion(pattern string, fused ir.OpType) {
	if capabilities.fusions[pattern] == nil {
		capabilities.fusions[pattern] = make(map[ir.OpType]bool)
	}

	capabilities.fusions[pattern][fused] = true
}

func (capabilities *StaticCapabilities) CanFuse(pattern string, fused ir.OpType) bool {
	return capabilities.fusions[pattern][fused]
}

func (capabilities *StaticCapabilities) SetShapeConstraints(
	operationID ir.OpType,
	constraints ...string,
) {
	capabilities.shapes[operationID] = slices.Clone(constraints)
}

func (capabilities *StaticCapabilities) ShapeConstraints(operationID ir.OpType) []string {
	return slices.Clone(capabilities.shapes[operationID])
}

func (capabilities *StaticCapabilities) Cost(node *ir.Node) Cost {
	elements := uint64(node.Shape().Len())
	if elements == 0 {
		elements = 1
	}

	switch node.OpType() {
	case ir.OpMatmul:
		dimensions := node.Shape().Dims()
		if len(dimensions) == 2 {
			return Cost{FLOPs: uint64(2 * dimensions[0] * dimensions[0] * dimensions[1])}
		}
	case ir.OpAdd, ir.OpMul, ir.OpReLU, ir.OpGELU, ir.OpTanh, ir.OpSigmoid:
		return Cost{FLOPs: elements, BytesRead: elements * 8, BytesWritten: elements * 8}
	}

	return Cost{FLOPs: elements, BytesRead: elements * 8, BytesWritten: elements * 8}
}
