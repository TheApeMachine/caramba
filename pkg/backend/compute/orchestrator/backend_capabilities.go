package orchestrator

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func CapabilitiesForLocation(location tensor.Location) Capabilities {
	switch location {
	case tensor.Host, tensor.CUDA, tensor.Metal, tensor.XLA:
		return fullBackendCapabilities(location)
	default:
		return NewDefaultCapabilities(location)
	}
}

func fullBackendCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)

	for _, operationID := range ir.RequiredOperationIDs() {
		capabilities.Register(operationID)
	}

	capabilities.RegisterFusion("matmul.activation", ir.OpFused)

	if location == tensor.Metal {
		capabilities.SetPrecision(tensor.Float32)
	}

	return capabilities
}
