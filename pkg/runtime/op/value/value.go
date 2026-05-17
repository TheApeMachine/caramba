package value

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
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
		return err
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

/*
Equals compares Inputs["left"] against either Inputs["right"] or
Config["right"] and writes the boolean result to Outputs["result"].
The Config form lets manifests compare against a literal without an
intermediate value.assign step (e.g. `right: "/exit"` for chat
command predicates). Comparison is via Go == for primitives and
ShouldResemble-shaped element-by-element matching for the supported
slice types ([]int, []float64, []string).
*/
type Equals struct{}

func (Equals) Execute(execContext op.Context) error {
	step := execContext.Step()
	leftRef, ok := step.Inputs["left"]

	if !ok {
		return fmt.Errorf("value.equals: missing input 'left'")
	}

	resultRef, ok := step.Outputs["result"]

	if !ok {
		return fmt.Errorf("value.equals: missing output 'result'")
	}

	leftValue, err := execContext.Resolve(leftRef)

	if err != nil {
		return err
	}

	rightValue, hasRight, err := equalsRightOperand(execContext, step)

	if err != nil {
		return err
	}

	if !hasRight {
		return fmt.Errorf("value.equals: missing right operand (inputs.right or config.right)")
	}

	return execContext.Bind(resultRef, equalsValues(leftValue, rightValue))
}

func equalsRightOperand(execContext op.Context, step program.Step) (any, bool, error) {
	if ref, ok := step.Inputs["right"]; ok {
		value, err := execContext.Resolve(ref)

		if err != nil {
			return nil, true, err
		}

		return value, true, nil
	}

	if raw, ok := step.Config["right"]; ok {
		return raw, true, nil
	}

	return nil, false, nil
}

func equalsValues(left, right any) bool {
	if left == nil || right == nil {
		return left == right
	}

	switch leftTyped := left.(type) {
	case []int:
		rightTyped, ok := right.([]int)

		if !ok || len(leftTyped) != len(rightTyped) {
			return false
		}

		for index, value := range leftTyped {
			if value != rightTyped[index] {
				return false
			}
		}

		return true
	case []float64:
		rightTyped, ok := right.([]float64)

		if !ok || len(leftTyped) != len(rightTyped) {
			return false
		}

		for index, value := range leftTyped {
			if value != rightTyped[index] {
				return false
			}
		}

		return true
	case []string:
		rightTyped, ok := right.([]string)

		if !ok || len(leftTyped) != len(rightTyped) {
			return false
		}

		for index, value := range leftTyped {
			if value != rightTyped[index] {
				return false
			}
		}

		return true
	}

	return left == right
}

func init() {
	op.Default.MustRegister("value.assign", Assign{})
	op.Default.MustRegister("value.append", Append{})
	op.Default.MustRegister("value.clear", Clear{})
	op.Default.MustRegister("value.slice", Slice{})
	op.Default.MustRegister("value.length", Length{})
	op.Default.MustRegister("value.positions", Positions{})
	op.Default.MustRegister("value.equals", Equals{})
}
