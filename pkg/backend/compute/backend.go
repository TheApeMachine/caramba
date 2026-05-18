package compute

import (
	"context"
	"errors"
	"fmt"

	computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	cpuoperation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	cpuoptimizer "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/orchestrator"
	"github.com/theapemachine/caramba/pkg/backend/compute/runner"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var ErrBackendRunnerRequired = errors.New("compute: backend runner is required")
var ErrDeviceRunnerUnavailable = errors.New("compute: dtype-native device graph runner is unavailable")

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
		return nil, fmt.Errorf("compute: cuda: %w", ErrDeviceRunnerUnavailable)
	case METAL:
		return nil, fmt.Errorf("compute: metal: %w", ErrDeviceRunnerUnavailable)
	case XLA:
		return nil, fmt.Errorf("compute: xla: %w", ErrDeviceRunnerUnavailable)
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
) (map[string]tensor.Tensor, error) {
	if backend == nil || backend.Runner == nil {
		return nil, ErrBackendRunnerRequired
	}

	scheduler := orchestrator.NewScheduler()
	scheduler.RegisterRunner(backend.Runner)

	return scheduler.Execute(ctx, graph, targets, backend.Runner.Location())
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
