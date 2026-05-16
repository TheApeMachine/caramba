package compute

import (
	"context"
	"errors"
	"fmt"

	computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	cpuoperation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	cpuoptimizer "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer"
	"github.com/theapemachine/caramba/pkg/backend/compute/cuda"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/metal"
	"github.com/theapemachine/caramba/pkg/backend/compute/runner"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/compute/xla"
)

var ErrBackendRunnerRequired = errors.New("compute: backend runner is required")

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
	Optimizers OptimizerRegistry
	Operations OperationRegistry
	Runner     runner.Runner
}

func NewBackend(backendType BackendType) (*Backend, error) {
	switch backendType {
	case CPU:
		return &Backend{
			Optimizers: NewOptimizerRegistry(cpuoptimizer.NewOptimizerRegistry()),
			Operations: NewOperationRegistry(cpuoperation.NewOperationRegistry()),
			Runner:     computecpu.NewRunner(),
		}, nil
	case CUDA:
		backend := &Backend{
			Optimizers: NewOptimizerRegistry(cuda.NewOptimizerRegistry()),
			Operations: NewOperationRegistry(cuda.NewOperationRegistry()),
			Runner:     cuda.NewRunner(),
		}

		return backend, backend.Available()
	case METAL:
		backend := &Backend{
			Optimizers: NewOptimizerRegistry(metal.NewOptimizerRegistry()),
			Operations: NewOperationRegistry(metal.NewOperationRegistry()),
			Runner:     metal.NewRunner(),
		}

		return backend, backend.Available()
	case XLA:
		backend := &Backend{
			Optimizers: NewOptimizerRegistry(xla.NewOptimizerRegistry()),
			Operations: NewOperationRegistry(xla.NewOperationRegistry()),
			Runner:     xla.NewRunner(),
		}

		return backend, backend.Available()
	}

	return nil, fmt.Errorf("compute: unsupported backend type %d", backendType)
}

func (backend *Backend) Available() error {
	if backend == nil || backend.Runner == nil {
		return ErrBackendRunnerRequired
	}

	availability, ok := backend.Runner.(interface {
		Err() error
	})

	if !ok {
		return nil
	}

	return availability.Err()
}

func (backend *Backend) Execute(
	ctx context.Context, graph *ir.Graph, targets []*ir.Node,
) (map[string]tensor.Float64Tensor, error) {
	if backend == nil || backend.Runner == nil {
		return nil, ErrBackendRunnerRequired
	}

	return backend.Runner.Execute(ctx, graph, targets)
}

func (backend *Backend) Location() tensor.Location {
	if backend == nil || backend.Runner == nil {
		return ""
	}

	return backend.Runner.Location()
}

func (backend *Backend) Close() error {
	if backend == nil || backend.Runner == nil {
		return nil
	}

	return backend.Runner.Close()
}
