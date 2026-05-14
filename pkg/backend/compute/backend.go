package compute

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer"
	"github.com/theapemachine/caramba/pkg/backend/compute/cuda"
	"github.com/theapemachine/caramba/pkg/backend/compute/metal"
	"github.com/theapemachine/caramba/pkg/backend/compute/xla"
)

type BackendType uint

const (
	CPU BackendType = iota
	METAL
	CUDA
	XLA
)

/*
Backend is the complete façade for hardware-local compute in Caramba: forward-graph execution,
scheduler capability accounting, and first-order optimisation on the matched Location.

Compliant implementations SHOULD wire ResidentGraphOperations, IrGraphRunner, and
OperationCapabilities onto one handle (or deterministic composition) plus a
StandardOptimizerRegistry that never crosses Location boundaries implicitly.
*/
type Backend struct {
	optimizers OptimizerRegistry
	operations OperationRegistry
}

func NewBackend(backendType BackendType) *Backend {
	switch backendType {
	case CPU:
		return &Backend{
			optimizers: NewOptimizerRegistry(optimizer.NewOptimizerRegistry()),
			operations: NewOperationRegistry(operation.NewOperationRegistry()),
		}
	case CUDA:
		return &Backend{
			optimizers: NewOptimizerRegistry(cuda.NewOptimizerRegistry()),
			operations: NewOperationRegistry(cuda.NewOperationRegistry()),
		}
	case METAL:
		return &Backend{
			optimizers: NewOptimizerRegistry(metal.NewOptimizerRegistry()),
			operations: NewOperationRegistry(metal.NewOperationRegistry()),
		}
	case XLA:
		return &Backend{
			optimizers: NewOptimizerRegistry(xla.NewOptimizerRegistry()),
			operations: NewOperationRegistry(xla.NewOperationRegistry()),
		}
	}

	return &Backend{
		optimizers: NewOptimizerRegistry(optimizer.NewOptimizerRegistry()),
		operations: NewOperationRegistry(operation.NewOperationRegistry()),
	}
}
