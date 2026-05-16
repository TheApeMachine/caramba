package orchestrator

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/dispatch"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func CapabilitiesForLocation(location tensor.Location) Capabilities {
	switch location {
	case tensor.Host:
		return hostCapabilities(location)
	case tensor.CUDA:
		return cudaCapabilities(location)
	case tensor.Metal:
		return metalCapabilities(location)
	case tensor.XLA:
		return xlaCapabilities(location)
	default:
		return NewDefaultCapabilities(location)
	}
}

func hostCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)

	for operationID := range dispatch.SupportedIDSet() {
		capabilities.Register(operationID)
	}

	capabilities.RegisterFusion("matmul.activation", ir.OpFused)

	return capabilities
}

func cudaCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)
	capabilities.Register(coreResidentOperationIDs()...)
	capabilities.RegisterFusion("matmul.activation", ir.OpFused)

	return capabilities
}

func metalCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)
	capabilities.Register(coreResidentOperationIDs()...)
	capabilities.Register(
		"embedding.token",
		"math.rmsnorm",
		"math.layernorm",
		"math.groupnorm",
		"shape.reshape",
		"shape.transpose",
		"shape.upsample_nearest2d",
		"shape.view_as_heads",
		"shape.merge_heads",
		"shape.last_token",
		"projection.linear",
		"attention.sdpa",
		"attention.gqa",
		"positional.rope",
		"convolution.conv2d",
		"convolution.conv_transpose2d",
	)
	capabilities.RegisterFusion("matmul.activation", ir.OpFused)
	capabilities.SetPrecision(tensor.Float32)

	return capabilities
}

func xlaCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)
	capabilities.Register(coreResidentOperationIDs()...)
	capabilities.RegisterFusion("matmul.activation", ir.OpFused)

	return capabilities
}

func coreResidentOperationIDs() []ir.OpType {
	return []ir.OpType{
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
		ir.OpSwish,
		ir.OpSELU,
		ir.OpFused,
		"math.add",
		"math.mul",
		"math.matmul",
		"activation.relu",
		"activation.leaky_relu",
		"activation.gelu",
		"activation.tanh",
		"activation.sigmoid",
		"activation.swiglu",
		"activation.swish",
		"activation.selu",
	}
}
