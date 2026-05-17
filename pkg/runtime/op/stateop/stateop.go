package stateop

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
Reset zeroes a declared state object via its Reset method.
*/
type ResetState struct{}

func (ResetState) Execute(execContext op.Context) error {
	target, ok := execContext.Step().Outputs["target"]

	if !ok {
		return fmt.Errorf("state.reset: missing output 'target'")
	}

	stateInstance, err := execContext.State(target.Name)

	if err != nil {
		return err
	}

	return stateInstance.Reset(execContext.Context())
}

/*
Append routes to the typed Append method on the destination state
object. Currently TokenBuffer is the supported sink; other state
types append through Update with a typed payload.
*/
type Append struct{}

func (Append) Execute(execContext op.Context) error {
	step := execContext.Step()
	target, ok := step.Outputs["target"]

	if !ok {
		return fmt.Errorf("state.append: missing output 'target'")
	}

	valueRef, ok := step.Inputs["value"]

	if !ok {
		return fmt.Errorf("state.append: missing input 'value'")
	}

	stateInstance, err := execContext.State(target.Name)

	if err != nil {
		return err
	}

	resolved, err := execContext.Resolve(valueRef)

	if err != nil {
		return err
	}

	return appendValue(stateInstance, resolved)
}

/*
Update applies a typed in-place update to a state object. Counter
supports Config["update"]: "increment" with optional Config["delta"];
other state types accept "set" with Inputs["value"].
*/
type Update struct{}

func (Update) Execute(execContext op.Context) error {
	step := execContext.Step()
	target, ok := step.Outputs["target"]

	if !ok {
		return fmt.Errorf("state.update: missing output 'target'")
	}

	stateInstance, err := execContext.State(target.Name)

	if err != nil {
		return err
	}

	updateName, _ := step.Config["update"].(string)

	switch counter := stateInstance.(type) {
	case *state.Counter:
		return updateCounter(execContext, counter, updateName, step.Config)
	}

	return fmt.Errorf(
		"state.update: type %q does not support update %q",
		stateInstance.Type(),
		updateName,
	)
}

/*
Inspect emits the named state object's Inspection record into
Outputs["report"] (a local value the program can subsequently log).
*/
type Inspect struct{}

func (Inspect) Execute(execContext op.Context) error {
	step := execContext.Step()
	target, ok := step.Inputs["target"]

	if !ok {
		return fmt.Errorf("state.inspect: missing input 'target'")
	}

	report, ok := step.Outputs["report"]

	if !ok {
		return fmt.Errorf("state.inspect: missing output 'report'")
	}

	stateInstance, err := execContext.State(target.Name)

	if err != nil {
		return err
	}

	inspection, err := stateInstance.Inspect(execContext.Context())

	if err != nil {
		return err
	}

	return execContext.Bind(report, inspection)
}

func appendValue(stateInstance state.State, value any) error {
	buffer, ok := stateInstance.(*state.TokenBuffer)

	if !ok {
		return fmt.Errorf(
			"state.append: type %q does not support append",
			stateInstance.Type(),
		)
	}

	switch typed := value.(type) {
	case int:
		buffer.Append(typed)

		return nil
	case int32:
		buffer.Append(int(typed))

		return nil
	case int64:
		buffer.Append(int(typed))

		return nil
	case []int:
		buffer.Extend(typed)

		return nil
	}

	return fmt.Errorf("state.append: cannot append %T to token_buffer", value)
}

func updateCounter(
	execContext op.Context, counter *state.Counter, update string, config map[string]any,
) error {
	switch update {
	case "", "increment":
		delta := 1

		if raw, ok := config["delta"]; ok {
			value, err := asInt(raw)

			if err != nil {
				return fmt.Errorf("state.update: counter delta: %w", err)
			}

			delta = value
		}

		counter.Increment(delta)

		return nil
	case "set":
		valueRef, ok := execContext.Step().Inputs["value"]

		if !ok {
			return fmt.Errorf("state.update: set requires inputs.value")
		}

		raw, err := execContext.Resolve(valueRef)

		if err != nil {
			return err
		}

		value, err := asInt(raw)

		if err != nil {
			return fmt.Errorf("state.update: counter set: %w", err)
		}

		counter.Set(value)

		return nil
	}

	return fmt.Errorf("state.update: counter does not support update %q", update)
}

func asInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func init() {
	op.Default.MustRegister("state.reset", ResetState{})
	op.Default.MustRegister("state.append", Append{})
	op.Default.MustRegister("state.update", Update{})
	op.Default.MustRegister("state.inspect", Inspect{})
}
