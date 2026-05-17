package telemetryops

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
EnterScope pushes a named scope onto the active telemetry stack.
Subsequent counter/histogram/tensor samples are tagged with the
current scope chain.
*/
type EnterScope struct{}

func (EnterScope) Execute(execContext op.Context) error {
	name, err := requiredString(execContext.Step().Config, "name")

	if err != nil {
		return fmt.Errorf("telemetry.scope.enter: %w", err)
	}

	execContext.Telemetry().EnterScope(name)

	return nil
}

/*
ExitScope pops the most recently entered telemetry scope.
*/
type ExitScope struct{}

func (ExitScope) Execute(execContext op.Context) error {
	execContext.Telemetry().ExitScope()

	return nil
}

/*
Counter increments a named telemetry counter. The delta comes from
Config["delta"] (default 1) or Inputs["delta"] when dynamic.
*/
type Counter struct{}

func (Counter) Execute(execContext op.Context) error {
	step := execContext.Step()
	name, err := requiredString(step.Config, "name")

	if err != nil {
		return fmt.Errorf("telemetry.counter: %w", err)
	}

	delta := 1.0

	if ref, ok := step.Inputs["delta"]; ok {
		value, err := execContext.Resolve(ref)

		if err != nil {
			return err
		}

		number, err := asFloat64(value)

		if err != nil {
			return fmt.Errorf("telemetry.counter: %w", err)
		}

		delta = number
	} else if value, ok := step.Config["delta"]; ok {
		number, err := asFloat64(value)

		if err != nil {
			return fmt.Errorf("telemetry.counter: %w", err)
		}

		delta = number
	}

	execContext.Telemetry().IncrementCounter(name, delta)

	return nil
}

/*
Histogram records one sample into a named histogram. The sample
comes from Inputs["value"] (resolved as float64).
*/
type Histogram struct{}

func (Histogram) Execute(execContext op.Context) error {
	step := execContext.Step()
	name, err := requiredString(step.Config, "name")

	if err != nil {
		return fmt.Errorf("telemetry.histogram: %w", err)
	}

	ref, ok := step.Inputs["value"]

	if !ok {
		return fmt.Errorf("telemetry.histogram: missing input 'value'")
	}

	raw, err := execContext.Resolve(ref)

	if err != nil {
		return err
	}

	value, err := asFloat64(raw)

	if err != nil {
		return fmt.Errorf("telemetry.histogram: %w", err)
	}

	execContext.Telemetry().RecordHistogram(name, value)

	return nil
}

/*
TraceTensor snapshots a tensor state's current values into the
telemetry stream so a researcher can replay activations after the
run finishes. The tensor must come from Inputs["tensor"].
*/
type TraceTensor struct{}

func (TraceTensor) Execute(execContext op.Context) error {
	step := execContext.Step()
	name, err := requiredString(step.Config, "name")

	if err != nil {
		return fmt.Errorf("telemetry.trace_tensor: %w", err)
	}

	ref, ok := step.Inputs["tensor"]

	if !ok {
		return fmt.Errorf("telemetry.trace_tensor: missing input 'tensor'")
	}

	raw, err := execContext.Resolve(ref)

	if err != nil {
		return err
	}

	values, shape, err := tensorValues(raw)

	if err != nil {
		return fmt.Errorf("telemetry.trace_tensor: %w", err)
	}

	execContext.Telemetry().RecordTensor(name, values, shape)

	return nil
}

func tensorValues(value any) ([]float64, []int, error) {
	switch typed := value.(type) {
	case *state.Tensor:
		return typed.Values(), typed.Shape(), nil
	case []float64:
		return typed, []int{len(typed)}, nil
	}

	return nil, nil, fmt.Errorf("expected tensor or []float64, got %T", value)
}

func requiredString(config map[string]any, key string) (string, error) {
	value, ok := config[key]

	if !ok {
		return "", fmt.Errorf("config.%s is required", key)
	}

	text, ok := value.(string)

	if !ok || text == "" {
		return "", fmt.Errorf("config.%s must be a non-empty string", key)
	}

	return text, nil
}

func asFloat64(value any) (float64, error) {
	switch typed := value.(type) {
	case float64:
		return typed, nil
	case float32:
		return float64(typed), nil
	case int:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	}

	return 0, fmt.Errorf("expected number, got %T", value)
}

func init() {
	op.Default.MustRegister("telemetry.scope.enter", EnterScope{})
	op.Default.MustRegister("telemetry.scope.exit", ExitScope{})
	op.Default.MustRegister("telemetry.counter", Counter{})
	op.Default.MustRegister("telemetry.histogram", Histogram{})
	op.Default.MustRegister("telemetry.trace_tensor", TraceTensor{})
}
