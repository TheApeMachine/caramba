package executor

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
Executor runs a compiled runtime program against a fixed set of
declared state, asset, sampler, scheduler, and graph instances.
External integrations (graph runner, sampler runner, scheduler
runner, tokenizer instances) are injected so the package depends
only on the runtime interfaces, not on the backend or chat trees.
*/
type Executor struct {
	program      *program.Program
	operations   *op.Registry
	states       map[string]state.State
	tokenizers   map[string]tokenizer.Tokenizer
	graphRunner  op.GraphRunner
	samplerRun   op.SamplerRunner
	schedulerRun op.SchedulerRunner
	telemetry    telemetry.Recorder
	stdin        io.Reader
	stdout       io.Writer
}

/*
Options configures an Executor at construction time. Only the
program is mandatory — the executor falls back to op.Default,
state.Default, os.Stdin/Stdout, and nil bridge runners when the
caller does not override them. Bridges that remain nil produce
explicit errors when the program asks for them, which is the only
way to keep the executor honest about what is wired up.
*/
type Options struct {
	Program         *program.Program
	Operations      *op.Registry
	StateRegistry   *state.Registry
	Tokenizers      map[string]tokenizer.Tokenizer
	GraphRunner     op.GraphRunner
	SamplerRunner   op.SamplerRunner
	SchedulerRunner op.SchedulerRunner
	Telemetry       telemetry.Recorder
	Stdin           io.Reader
	Stdout          io.Writer
}

/*
New constructs an Executor and instantiates every declared state
object via the supplied state registry. The Executor is ready to
Run once this returns.
*/
func New(options Options) (*Executor, error) {
	if options.Program == nil {
		return nil, fmt.Errorf("runtime/executor: program is required")
	}

	if err := options.Program.Validate(); err != nil {
		return nil, fmt.Errorf("runtime/executor: program failed validation: %w", err)
	}

	registry := options.Operations

	if registry == nil {
		registry = op.Default
	}

	stateRegistry := options.StateRegistry

	if stateRegistry == nil {
		stateRegistry = state.Default
	}

	states, err := buildStates(options.Program, stateRegistry)

	if err != nil {
		return nil, err
	}

	executor := &Executor{
		program:      options.Program,
		operations:   registry,
		states:       states,
		tokenizers:   options.Tokenizers,
		graphRunner:  options.GraphRunner,
		samplerRun:   options.SamplerRunner,
		schedulerRun: options.SchedulerRunner,
		telemetry:    options.Telemetry,
		stdin:        options.Stdin,
		stdout:       options.Stdout,
	}

	return executor, nil
}

/*
Run executes the program once. It returns the first non-control
error any operation produces; ErrBreak/ErrContinue raised at the
top level are reported as misuse since they only mean something
inside a loop body.
*/
func (executor *Executor) Run(runContext context.Context) error {
	scope := NewScope()

	return executor.runSteps(runContext, executor.program.Steps, scope)
}

/*
States returns the live state instances keyed by id. The runtime
inspection surface consumes this directly.
*/
func (executor *Executor) States() map[string]state.State {
	out := make(map[string]state.State, len(executor.states))

	for stateID, instance := range executor.states {
		out[stateID] = instance
	}

	return out
}

func (executor *Executor) runSteps(
	runContext context.Context, steps []program.Step, scope *Scope,
) error {
	for index := range steps {
		if err := runContext.Err(); err != nil {
			return err
		}

		err := executor.runStep(runContext, &steps[index], scope)

		if err == nil {
			continue
		}

		if errors.Is(err, op.ErrBreak) || errors.Is(err, op.ErrContinue) {
			return err
		}

		return fmt.Errorf("step %q (%s): %w", steps[index].ID, steps[index].Op, err)
	}

	return nil
}

func (executor *Executor) runStep(
	runContext context.Context, step *program.Step, scope *Scope,
) error {
	operation, err := executor.operations.Lookup(step.Op)

	if err != nil {
		return err
	}

	stepContext := &runtimeContext{
		executor:    executor,
		runContext:  runContext,
		currentStep: *step,
		scope:       scope,
	}

	return operation.Execute(stepContext)
}

func buildStates(
	runtimeProgram *program.Program, registry *state.Registry,
) (map[string]state.State, error) {
	instances := make(map[string]state.State, len(runtimeProgram.State))

	for _, declaration := range runtimeProgram.State {
		instance, err := registry.Build(declaration.Type, declaration.ID, declaration.Config)

		if err != nil {
			return nil, fmt.Errorf("runtime/executor: building state %q: %w", declaration.ID, err)
		}

		instances[declaration.ID] = instance
	}

	return instances, nil
}
