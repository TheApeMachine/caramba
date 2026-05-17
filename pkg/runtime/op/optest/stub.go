package optest

import (
	"bytes"
	"context"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
StubContext is the simplest possible op.Context used by op subpackage
unit tests. It carries a flat local variable scope, a map of state
objects, asset/sampler/scheduler/graph declarations, and a body
runner closure injected by the test. The stub deliberately avoids
hierarchical scopes — tests that need scoping should drive through
the real executor.
*/
type StubContext struct {
	StepRef     program.Step
	Ctx         context.Context
	Scope       map[string]any
	States      map[string]state.State
	Assets      map[string]program.AssetDeclaration
	Samplers    map[string]program.SamplerDeclaration
	Schedulers  map[string]program.SchedulerDeclaration
	Graphs      map[string]program.GraphModule
	Tokenizers  map[string]tokenizer.Tokenizer
	StdinBuf    *bytes.Buffer
	StdoutBuf   *bytes.Buffer
	BodyHandler func([]program.Step) error
	Graphs_     op.GraphRunner
	Samplers_   op.SamplerRunner
	Schedulers_ op.SchedulerRunner
	Recorder    telemetry.Recorder
}

func NewStubContext() *StubContext {
	return &StubContext{
		Ctx:        context.Background(),
		Scope:      map[string]any{},
		States:     map[string]state.State{},
		Assets:     map[string]program.AssetDeclaration{},
		Samplers:   map[string]program.SamplerDeclaration{},
		Schedulers: map[string]program.SchedulerDeclaration{},
		Graphs:     map[string]program.GraphModule{},
		Tokenizers: map[string]tokenizer.Tokenizer{},
		StdinBuf:   &bytes.Buffer{},
		StdoutBuf:  &bytes.Buffer{},
	}
}

func (stubContext *StubContext) Context() context.Context {
	return stubContext.Ctx
}

func (stubContext *StubContext) Step() program.Step {
	return stubContext.StepRef
}

func (stubContext *StubContext) Resolve(ref program.ValueRef) (any, error) {
	switch ref.Namespace {
	case program.NamespaceLocal:
		value, ok := stubContext.Scope[ref.Name]

		if !ok {
			return nil, fmt.Errorf("stub: local %q is not bound", ref.Name)
		}

		return value, nil
	case program.NamespaceLiteral:
		return ref.Name, nil
	case program.NamespaceState:
		return stubContext.State(ref.Name)
	case program.NamespaceAsset:
		return stubContext.Asset(ref.Name)
	case program.NamespaceSampler:
		return stubContext.Sampler(ref.Name)
	case program.NamespaceScheduler:
		return stubContext.Scheduler(ref.Name)
	case program.NamespaceGraph:
		return stubContext.Graph(ref.Name)
	case program.NamespaceTokenizer:
		return stubContext.Tokenizer(ref.Name)
	}

	return nil, fmt.Errorf("stub: unknown namespace %q", ref.Namespace)
}

func (stubContext *StubContext) Bind(ref program.ValueRef, value any) error {
	if ref.Namespace != program.NamespaceLocal {
		return fmt.Errorf("stub: cannot bind to namespace %q", ref.Namespace)
	}

	stubContext.Scope[ref.Name] = value

	return nil
}

func (stubContext *StubContext) State(stateID string) (state.State, error) {
	value, ok := stubContext.States[stateID]

	if !ok {
		return nil, fmt.Errorf("stub: state %q is not declared", stateID)
	}

	return value, nil
}

func (stubContext *StubContext) Asset(assetID string) (program.AssetDeclaration, error) {
	value, ok := stubContext.Assets[assetID]

	if !ok {
		return program.AssetDeclaration{}, fmt.Errorf("stub: asset %q is not declared", assetID)
	}

	return value, nil
}

func (stubContext *StubContext) Sampler(samplerID string) (program.SamplerDeclaration, error) {
	value, ok := stubContext.Samplers[samplerID]

	if !ok {
		return program.SamplerDeclaration{}, fmt.Errorf("stub: sampler %q is not declared", samplerID)
	}

	return value, nil
}

func (stubContext *StubContext) Scheduler(schedulerID string) (program.SchedulerDeclaration, error) {
	value, ok := stubContext.Schedulers[schedulerID]

	if !ok {
		return program.SchedulerDeclaration{}, fmt.Errorf("stub: scheduler %q is not declared", schedulerID)
	}

	return value, nil
}

func (stubContext *StubContext) Graph(graphID string) (program.GraphModule, error) {
	value, ok := stubContext.Graphs[graphID]

	if !ok {
		return program.GraphModule{}, fmt.Errorf("stub: graph %q is not declared", graphID)
	}

	return value, nil
}

func (stubContext *StubContext) Tokenizer(assetID string) (tokenizer.Tokenizer, error) {
	value, ok := stubContext.Tokenizers[assetID]

	if !ok {
		return nil, fmt.Errorf("stub: tokenizer %q is not wired", assetID)
	}

	return value, nil
}

func (stubContext *StubContext) Run(steps []program.Step) error {
	if stubContext.BodyHandler == nil {
		return fmt.Errorf("stub: BodyHandler is not configured")
	}

	return stubContext.BodyHandler(steps)
}

func (stubContext *StubContext) RunBody(steps []program.Step) error {
	return stubContext.Run(steps)
}

func (stubContext *StubContext) Stdin() io.Reader {
	return stubContext.StdinBuf
}

func (stubContext *StubContext) Stdout() io.Writer {
	return stubContext.StdoutBuf
}

func (stubContext *StubContext) GraphRunner() op.GraphRunner {
	return stubContext.Graphs_
}

func (stubContext *StubContext) SamplerRunner() op.SamplerRunner {
	return stubContext.Samplers_
}

func (stubContext *StubContext) SchedulerRunner() op.SchedulerRunner {
	return stubContext.Schedulers_
}

func (stubContext *StubContext) Telemetry() telemetry.Recorder {
	if stubContext.Recorder == nil {
		return telemetry.NoOp{}
	}

	return stubContext.Recorder
}
