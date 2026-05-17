package executor

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
runtimeContext is the per-step implementation of op.Context. The
executor builds one for every dispatched step; ops do not see the
Executor struct directly so they cannot reach past the interface.
*/
type runtimeContext struct {
	executor    *Executor
	runContext  context.Context
	currentStep program.Step
	scope       *Scope
	bodyScope   *Scope
}

func (rc *runtimeContext) Context() context.Context {
	return rc.runContext
}

func (rc *runtimeContext) Step() program.Step {
	return rc.currentStep
}

/*
Resolve turns a ValueRef into a concrete value the op can use. The
namespace decides where the lookup goes; local values come from the
scope, declared objects come from their registries, asset/sampler/
scheduler/graph references return their declarations so the op can
hand them to the appropriate runner.
*/
func (rc *runtimeContext) Resolve(ref program.ValueRef) (any, error) {
	if ref.Namespace != program.NamespaceLiteral && ref.Name == "" {
		return nil, fmt.Errorf("runtime/executor: empty ref.Name for namespace %q", ref.Namespace)
	}

	switch ref.Namespace {
	case program.NamespaceLocal:
		return rc.scope.Get(ref.Name)
	case program.NamespaceLiteral:
		return ref.Name, nil
	case program.NamespaceState:
		return rc.State(ref.Name)
	case program.NamespaceAsset:
		return rc.Asset(ref.Name)
	case program.NamespaceSampler:
		return rc.Sampler(ref.Name)
	case program.NamespaceScheduler:
		return rc.Scheduler(ref.Name)
	case program.NamespaceGraph:
		return rc.Graph(ref.Name)
	case program.NamespaceTokenizer:
		return rc.Tokenizer(ref.Name)
	}

	return nil, fmt.Errorf("runtime/executor: cannot resolve namespace %q", ref.Namespace)
}

/*
Bind writes a value into the destination ref. Local refs write to
the current scope; state refs replace the state instance entirely
(used by ops that rebuild a state). Other namespaces are immutable
during execution.
*/
func (rc *runtimeContext) Bind(ref program.ValueRef, value any) error {
	switch ref.Namespace {
	case program.NamespaceLocal:
		rc.scope.Set(ref.Name, value)

		return nil
	case program.NamespaceState:
		newState, ok := value.(state.State)

		if !ok {
			return fmt.Errorf(
				"runtime/executor: state %q can only be bound to a state.State, got %T",
				ref.Name,
				value,
			)
		}

		rc.executor.states[ref.Name] = newState

		return nil
	}

	return fmt.Errorf(
		"runtime/executor: namespace %q is read-only at runtime",
		ref.Namespace,
	)
}

func (rc *runtimeContext) State(stateID string) (state.State, error) {
	instance, ok := rc.executor.states[stateID]

	if !ok {
		return nil, fmt.Errorf("runtime/executor: state %q is not instantiated", stateID)
	}

	return instance, nil
}

func (rc *runtimeContext) Asset(assetID string) (program.AssetDeclaration, error) {
	declaration := rc.executor.program.AssetByID(assetID)

	if declaration == nil {
		return program.AssetDeclaration{}, fmt.Errorf(
			"runtime/executor: asset %q is not declared",
			assetID,
		)
	}

	return *declaration, nil
}

func (rc *runtimeContext) Sampler(samplerID string) (program.SamplerDeclaration, error) {
	declaration := rc.executor.program.SamplerByID(samplerID)

	if declaration == nil {
		return program.SamplerDeclaration{}, fmt.Errorf(
			"runtime/executor: sampler %q is not declared",
			samplerID,
		)
	}

	return *declaration, nil
}

func (rc *runtimeContext) Scheduler(schedulerID string) (program.SchedulerDeclaration, error) {
	declaration := rc.executor.program.SchedulerByID(schedulerID)

	if declaration == nil {
		return program.SchedulerDeclaration{}, fmt.Errorf(
			"runtime/executor: scheduler %q is not declared",
			schedulerID,
		)
	}

	return *declaration, nil
}

func (rc *runtimeContext) Graph(graphID string) (program.GraphModule, error) {
	module, ok := rc.executor.program.Graphs[graphID]

	if !ok {
		return program.GraphModule{}, fmt.Errorf(
			"runtime/executor: graph %q is not declared",
			graphID,
		)
	}

	if module.ID == "" {
		module.ID = graphID
	}

	return module, nil
}

func (rc *runtimeContext) Tokenizer(assetID string) (tokenizer.Tokenizer, error) {
	if rc.executor.tokenizers == nil {
		return nil, fmt.Errorf("runtime/executor: no tokenizer instances are wired up")
	}

	instance, ok := rc.executor.tokenizers[assetID]

	if !ok {
		return nil, fmt.Errorf("runtime/executor: tokenizer %q is not wired up", assetID)
	}

	return instance, nil
}

/*
Run executes a nested body of steps in a fresh child scope. Each
call creates a new scope so successive invocations are isolated.
Use this when the body should not share bindings with sibling
invocations.
*/
func (rc *runtimeContext) Run(steps []program.Step) error {
	child := rc.scope.Child()

	return rc.executor.runSteps(rc.runContext, steps, child)
}

/*
RunBody is like Run but uses a single child scope that persists
across every call within the same op.Execute invocation. Loop ops
(loop_count, loop_each, loop_until) call RunBody once per iteration
so writes from one iteration's body (notably value.assign carries
like chat's carry_token → input_ids) are visible to the next.
The persistent scope is discarded when the outer op.Execute returns.
*/
func (rc *runtimeContext) RunBody(steps []program.Step) error {
	if rc.bodyScope == nil {
		rc.bodyScope = rc.scope.Child()
	}

	return rc.executor.runSteps(rc.runContext, steps, rc.bodyScope)
}

func (rc *runtimeContext) Stdin() io.Reader {
	if rc.executor.stdin == nil {
		return os.Stdin
	}

	return rc.executor.stdin
}

func (rc *runtimeContext) Stdout() io.Writer {
	if rc.executor.stdout == nil {
		return os.Stdout
	}

	return rc.executor.stdout
}

func (rc *runtimeContext) GraphRunner() op.GraphRunner {
	return rc.executor.graphRunner
}

func (rc *runtimeContext) SamplerRunner() op.SamplerRunner {
	return rc.executor.samplerRun
}

func (rc *runtimeContext) SchedulerRunner() op.SchedulerRunner {
	return rc.executor.schedulerRun
}

func (rc *runtimeContext) Telemetry() telemetry.Recorder {
	if rc.executor.telemetry == nil {
		return telemetry.NoOp{}
	}

	return rc.executor.telemetry
}
