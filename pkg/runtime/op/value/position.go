package value

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
Length writes the element count of Inputs["value"] to Outputs["length"].
Scalars count as one.
*/
type Length struct{}

func (Length) Execute(execContext op.Context) error {
	step := execContext.Step()
	valueRef, ok := step.Inputs["value"]

	if !ok {
		return fmt.Errorf("value.length: missing input 'value'")
	}

	outputRef, ok := step.Outputs["length"]

	if !ok {
		return fmt.Errorf("value.length: missing output 'length'")
	}

	value, err := execContext.Resolve(valueRef)

	if err != nil {
		return err
	}

	length, err := lengthOf(value)

	if err != nil {
		return fmt.Errorf("value.length: %w", err)
	}

	return execContext.Bind(outputRef, length)
}

/*
Positions creates contiguous token positions starting at Inputs["start"].
The number of positions equals the length of Inputs["tokens"].
*/
type Positions struct{}

func (Positions) Execute(execContext op.Context) error {
	step := execContext.Step()
	tokensRef, ok := step.Inputs["tokens"]

	if !ok {
		return fmt.Errorf("value.positions: missing input 'tokens'")
	}

	outputRef, ok := step.Outputs["positions"]

	if !ok {
		return fmt.Errorf("value.positions: missing output 'positions'")
	}

	start, err := positionStart(execContext)

	if err != nil {
		return err
	}

	tokens, err := execContext.Resolve(tokensRef)

	if err != nil {
		return err
	}

	length, err := lengthOf(tokens)

	if err != nil {
		return fmt.Errorf("value.positions: %w", err)
	}

	positions := make([]int, length)

	for index := range positions {
		positions[index] = start + index
	}

	return execContext.Bind(outputRef, positions)
}

func positionStart(execContext op.Context) (int, error) {
	startRef, ok := execContext.Step().Inputs["start"]

	if !ok {
		return 0, nil
	}

	raw, err := execContext.Resolve(startRef)

	if err != nil {
		return 0, err
	}

	switch typed := raw.(type) {
	case *state.Counter:
		return typed.Value(), nil
	default:
		value, err := asInt(raw)

		if err != nil {
			return 0, fmt.Errorf("value.positions: start: %w", err)
		}

		return value, nil
	}
}

func lengthOf(value any) (int, error) {
	switch typed := value.(type) {
	case []any:
		return len(typed), nil
	case []int:
		return len(typed), nil
	case []float64:
		return len(typed), nil
	case []string:
		return len(typed), nil
	case int, int32, int64, float64:
		return 1, nil
	case *state.Counter:
		return 1, nil
	case *state.Tensor:
		return len(typed.Values()), nil
	case *state.TokenBuffer:
		return typed.Length(), nil
	}

	return 0, fmt.Errorf("unsupported value type %T", value)
}
