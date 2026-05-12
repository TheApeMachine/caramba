package orchestrator

import (
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
	CanFuse(pattern string, fused ir.OpType) bool
	Cost(node *ir.Node) Cost
}

type CapabilityProvider interface {
	Capabilities() Capabilities
}

type StaticCapabilities struct {
	location tensor.Location
	ops      map[ir.OpType]bool
	fusions  map[string]map[ir.OpType]bool
}

func NewStaticCapabilities(location tensor.Location) *StaticCapabilities {
	return &StaticCapabilities{
		location: location,
		ops:      make(map[ir.OpType]bool),
		fusions:  make(map[string]map[ir.OpType]bool),
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

func (capabilities *StaticCapabilities) RegisterFusion(pattern string, fused ir.OpType) {
	if capabilities.fusions[pattern] == nil {
		capabilities.fusions[pattern] = make(map[ir.OpType]bool)
	}

	capabilities.fusions[pattern][fused] = true
}

func (capabilities *StaticCapabilities) CanFuse(pattern string, fused ir.OpType) bool {
	return capabilities.fusions[pattern][fused]
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
