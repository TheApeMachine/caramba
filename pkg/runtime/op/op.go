package op

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"sync"

	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
Operation is the contract every runtime op satisfies. The executor
holds a registry keyed by program.OperationID; each step in the
runtime IR is dispatched to the registered Operation.

Operations are stateless. All per-run state lives behind the Context
they receive: the variable scope, declared state objects, declared
assets, the tokenizer, the I/O channels, and the executor's body
runner for nested steps.
*/
type Operation interface {
	Execute(execContext Context) error
}

/*
Context is the surface ops use to read and mutate the running
program. It is implemented by the executor; the op packages depend
only on this interface so they can be tested without spinning the
whole executor up.
*/
type Context interface {
	Context() context.Context
	Step() program.Step
	Resolve(ref program.ValueRef) (any, error)
	Bind(ref program.ValueRef, value any) error
	State(stateID string) (state.State, error)
	Asset(assetID string) (program.AssetDeclaration, error)
	Sampler(samplerID string) (program.SamplerDeclaration, error)
	Scheduler(schedulerID string) (program.SchedulerDeclaration, error)
	Graph(graphID string) (program.GraphModule, error)
	Run(steps []program.Step) error
	RunBody(steps []program.Step) error
	Stdin() io.Reader
	Stdout() io.Writer
	Tokenizer(assetID string) (tokenizer.Tokenizer, error)
	GraphRunner() GraphRunner
	SamplerRunner() SamplerRunner
	SchedulerRunner() SchedulerRunner
	Telemetry() telemetry.Recorder
}

/*
GraphRunner is the bridge to pkg/backend/compute that the graph.call
op uses. The executor holds the concrete implementation; ops never
import the backend directly.
*/
type GraphRunner interface {
	Call(
		execContext context.Context,
		graph program.GraphModule,
		inputs map[string]any,
	) (map[string]any, error)
}

/*
SamplerRunner advances a declared sampler given a logits tensor and
the current decode history. The executor wires this to the concrete
sampler implementations.
*/
type SamplerRunner interface {
	Next(
		execContext context.Context,
		sampler program.SamplerDeclaration,
		logits []float64,
		history []int,
	) (token int, stopped bool, err error)
}

/*
SchedulerRunner advances a declared diffusion scheduler one step.
*/
type SchedulerRunner interface {
	Timesteps(
		execContext context.Context,
		scheduler program.SchedulerDeclaration,
	) ([]float64, error)
	Step(
		execContext context.Context,
		scheduler program.SchedulerDeclaration,
		stepIndex int,
		latents []float64,
		modelOutput []float64,
	) ([]float64, error)
}

/*
ErrBreak and ErrContinue are sentinel signals control-flow ops emit
to short-circuit loop bodies. The executor watches for them; nothing
else should treat them as real errors.
*/
var (
	ErrBreak    = errors.New("runtime/op: break")
	ErrContinue = errors.New("runtime/op: continue")
)

/*
Registry maps program.OperationID to Operation. The default registry
is populated by the various op subpackages in their init functions.
*/
type Registry struct {
	mu         sync.RWMutex
	operations map[program.OperationID]Operation
}

func NewRegistry() *Registry {
	return &Registry{operations: map[program.OperationID]Operation{}}
}

func (registry *Registry) Register(operationID program.OperationID, operation Operation) error {
	registry.mu.Lock()
	defer registry.mu.Unlock()

	if _, exists := registry.operations[operationID]; exists {
		return fmt.Errorf("runtime/op: operation %q already registered", operationID)
	}

	registry.operations[operationID] = operation

	return nil
}

func (registry *Registry) MustRegister(operationID program.OperationID, operation Operation) {
	if err := registry.Register(operationID, operation); err != nil {
		panic(err)
	}
}

func (registry *Registry) Lookup(operationID program.OperationID) (Operation, error) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	operation, ok := registry.operations[operationID]

	if !ok {
		return nil, fmt.Errorf("runtime/op: operation %q is not registered", operationID)
	}

	return operation, nil
}

func (registry *Registry) IDs() []program.OperationID {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	ids := make([]program.OperationID, 0, len(registry.operations))

	for operationID := range registry.operations {
		ids = append(ids, operationID)
	}

	sort.Slice(ids, func(i, j int) bool { return string(ids[i]) < string(ids[j]) })

	return ids
}

/*
Default is the runtime-wide registry. Each op subpackage registers
itself onto Default in its init function so the executor only has to
construct one set of factories.
*/
var Default = NewRegistry()
