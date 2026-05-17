package control

import (
	"errors"
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
LoopCount runs its body Config["count"] times. Count must resolve
to a positive integer either as a literal in Config or by resolving
a value reference also stored in Config under "count_from".
*/
type LoopCount struct{}

func (LoopCount) Execute(execContext op.Context) error {
	count, err := resolveCount(execContext)

	if err != nil {
		return err
	}

	for iteration := 0; iteration < count; iteration++ {
		bodyErr := execContext.Run(execContext.Step().Body)

		if bodyErr == nil {
			continue
		}

		if errors.Is(bodyErr, op.ErrBreak) {
			return nil
		}

		if errors.Is(bodyErr, op.ErrContinue) {
			continue
		}

		return bodyErr
	}

	return nil
}

/*
LoopEach iterates a slice value and binds each element to a local
named by Config["as"] before executing the body.
*/
type LoopEach struct{}

func (LoopEach) Execute(execContext op.Context) error {
	step := execContext.Step()
	binding, ok := step.Config["as"].(string)

	if !ok || binding == "" {
		return fmt.Errorf("control.loop_each: config.as must be a non-empty string")
	}

	source, ok := step.Inputs["source"]

	if !ok {
		return fmt.Errorf("control.loop_each: missing input 'source'")
	}

	value, err := execContext.Resolve(source)

	if err != nil {
		return err
	}

	values, err := toAnySlice(value)

	if err != nil {
		return fmt.Errorf("control.loop_each: %w", err)
	}

	bindingRef := program.ValueRef{Namespace: program.NamespaceLocal, Name: binding}

	for _, element := range values {
		if err := execContext.Bind(bindingRef, element); err != nil {
			return err
		}

		bodyErr := execContext.Run(step.Body)

		if bodyErr == nil {
			continue
		}

		if errors.Is(bodyErr, op.ErrBreak) {
			return nil
		}

		if errors.Is(bodyErr, op.ErrContinue) {
			continue
		}

		return bodyErr
	}

	return nil
}

/*
LoopUntil runs the body until the value referenced by Inputs["condition"]
resolves to true. A maximum iteration cap from Config["max"] guards
against accidental infinite loops; zero means unlimited.
*/
type LoopUntil struct{}

func (LoopUntil) Execute(execContext op.Context) error {
	step := execContext.Step()
	conditionRef, ok := step.Inputs["condition"]

	if !ok {
		return fmt.Errorf("control.loop_until: missing input 'condition'")
	}

	maxIterations, err := intFromConfig(step.Config, "max")

	if err != nil {
		return fmt.Errorf("control.loop_until: %w", err)
	}

	iteration := 0

	for {
		// maxIterations counts body executions, not condition checks. A
		// value of N permits exactly N body runs before erroring; the
		// guard runs at the top of the loop so an unsatisfied condition
		// after the Nth body invocation surfaces the error here.
		if maxIterations > 0 && iteration >= maxIterations {
			return fmt.Errorf(
				"control.loop_until: exceeded max iterations %d without satisfying condition",
				maxIterations,
			)
		}

		bodyErr := execContext.Run(step.Body)

		if errors.Is(bodyErr, op.ErrBreak) {
			return nil
		}

		if bodyErr != nil && !errors.Is(bodyErr, op.ErrContinue) {
			return bodyErr
		}

		conditionValue, err := execContext.Resolve(conditionRef)

		if err != nil {
			return err
		}

		satisfied, err := asBool(conditionValue)

		if err != nil {
			return fmt.Errorf("control.loop_until: %w", err)
		}

		if satisfied {
			return nil
		}

		iteration++
	}
}

/*
BreakIf raises op.ErrBreak when its condition input is truthy.
*/
type BreakIf struct{}

func (BreakIf) Execute(execContext op.Context) error {
	step := execContext.Step()
	ref, ok := step.Inputs["condition"]

	if !ok {
		return fmt.Errorf("control.break_if: missing input 'condition'")
	}

	value, err := execContext.Resolve(ref)

	if err != nil {
		return err
	}

	satisfied, err := asBool(value)

	if err != nil {
		return fmt.Errorf("control.break_if: %w", err)
	}

	if satisfied {
		return op.ErrBreak
	}

	return nil
}

/*
ContinueIf raises op.ErrContinue when its condition input is truthy.
*/
type ContinueIf struct{}

func (ContinueIf) Execute(execContext op.Context) error {
	step := execContext.Step()
	ref, ok := step.Inputs["condition"]

	if !ok {
		return fmt.Errorf("control.continue_if: missing input 'condition'")
	}

	value, err := execContext.Resolve(ref)

	if err != nil {
		return err
	}

	satisfied, err := asBool(value)

	if err != nil {
		return fmt.Errorf("control.continue_if: %w", err)
	}

	if satisfied {
		return op.ErrContinue
	}

	return nil
}

func init() {
	op.Default.MustRegister("control.loop_count", LoopCount{})
	op.Default.MustRegister("control.loop_each", LoopEach{})
	op.Default.MustRegister("control.loop_until", LoopUntil{})
	op.Default.MustRegister("control.break_if", BreakIf{})
	op.Default.MustRegister("control.continue_if", ContinueIf{})
}
