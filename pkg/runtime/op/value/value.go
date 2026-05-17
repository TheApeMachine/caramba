package value

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
)

/*
Assign reads Inputs["value"] and writes it to Outputs["target"].
This is the runtime equivalent of a simple `=` assignment.
*/
type Assign struct{}

func (Assign) Execute(execContext op.Context) error {
	step := execContext.Step()
	source, ok := step.Inputs["value"]

	if !ok {
		return fmt.Errorf("value.assign: missing input 'value'")
	}

	target, ok := step.Outputs["target"]

	if !ok {
		return fmt.Errorf("value.assign: missing output 'target'")
	}

	resolved, err := execContext.Resolve(source)

	if err != nil {
		return err
	}

	return execContext.Bind(target, resolved)
}

/*
Append concatenates an element onto the end of a list-shaped local
value, writing the result back to the same target. The current
target value must be a []any, []int, []float64, or []string slice.
*/
type Append struct{}

func (Append) Execute(execContext op.Context) error {
	step := execContext.Step()
	targetRef, ok := step.Outputs["target"]

	if !ok {
		return fmt.Errorf("value.append: missing output 'target'")
	}

	elementRef, ok := step.Inputs["element"]

	if !ok {
		return fmt.Errorf("value.append: missing input 'element'")
	}

	currentValue, err := execContext.Resolve(targetRef)

	if err != nil {
		return err
	}

	element, err := execContext.Resolve(elementRef)

	if err != nil {
		return err
	}

	next, err := appendAny(currentValue, element)

	if err != nil {
		return err
	}

	return execContext.Bind(targetRef, next)
}

/*
Clear replaces a local value with the zero value of its slice type.
*/
type Clear struct{}

func (Clear) Execute(execContext op.Context) error {
	step := execContext.Step()
	targetRef, ok := step.Outputs["target"]

	if !ok {
		return fmt.Errorf("value.clear: missing output 'target'")
	}

	currentValue, err := execContext.Resolve(targetRef)

	if err != nil {
		return execContext.Bind(targetRef, []any{})
	}

	switch currentValue.(type) {
	case []int:
		return execContext.Bind(targetRef, []int{})
	case []float64:
		return execContext.Bind(targetRef, []float64{})
	case []string:
		return execContext.Bind(targetRef, []string{})
	}

	return execContext.Bind(targetRef, []any{})
}

/*
Slice extracts [start:end) from a slice value. Negative indices are
not supported; missing end means "to the end of the slice".
*/
type Slice struct{}

func (Slice) Execute(execContext op.Context) error {
	step := execContext.Step()
	sourceRef, ok := step.Inputs["source"]

	if !ok {
		return fmt.Errorf("value.slice: missing input 'source'")
	}

	targetRef, ok := step.Outputs["target"]

	if !ok {
		return fmt.Errorf("value.slice: missing output 'target'")
	}

	source, err := execContext.Resolve(sourceRef)

	if err != nil {
		return err
	}

	start, err := intFromConfig(step.Config, "start")

	if err != nil {
		return fmt.Errorf("value.slice: %w", err)
	}

	end, err := intFromConfigOrLen(step.Config, "end", source)

	if err != nil {
		return fmt.Errorf("value.slice: %w", err)
	}

	sliced, err := sliceAny(source, start, end)

	if err != nil {
		return err
	}

	return execContext.Bind(targetRef, sliced)
}

func init() {
	op.Default.MustRegister("value.assign", Assign{})
	op.Default.MustRegister("value.append", Append{})
	op.Default.MustRegister("value.clear", Clear{})
	op.Default.MustRegister("value.slice", Slice{})
}
