package graph

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
Call invokes a declared graph module through the executor's
GraphRunner. The runtime program references the target graph via
Inputs["graph"] (NamespaceGraph). All remaining entries in Inputs
are passed through as named tensor / state inputs. Each entry in
Outputs receives the corresponding returned value.
*/
type Call struct{}

func (Call) Execute(execContext op.Context) error {
	step := execContext.Step()

	runner := execContext.GraphRunner()

	if runner == nil {
		return fmt.Errorf("graph.call: no GraphRunner is wired up")
	}

	graphRef, ok := step.Inputs["graph"]

	if !ok || graphRef.Namespace != program.NamespaceGraph {
		return fmt.Errorf("graph.call: inputs.graph must reference the graph namespace")
	}

	module, err := execContext.Graph(graphRef.Name)

	if err != nil {
		return err
	}

	inputs, err := resolveInputs(execContext, step.Inputs)

	if err != nil {
		return fmt.Errorf("graph.call: %w", err)
	}

	outputs, err := runner.Call(execContext.Context(), module, inputs)

	if err != nil {
		return fmt.Errorf("graph.call: %w", err)
	}

	return bindOutputs(execContext, step.Outputs, outputs)
}

func init() {
	op.Default.MustRegister("graph.call", Call{})
}

func resolveInputs(
	execContext op.Context, refs map[string]program.ValueRef,
) (map[string]any, error) {
	resolved := make(map[string]any, len(refs))

	for name, ref := range refs {
		if name == "graph" {
			continue
		}

		value, err := execContext.Resolve(ref)

		if err != nil {
			return nil, fmt.Errorf("input %q: %w", name, err)
		}

		resolved[name] = value
	}

	return resolved, nil
}

func bindOutputs(
	execContext op.Context, refs map[string]program.ValueRef, outputs map[string]any,
) error {
	for name, ref := range refs {
		value, ok := outputs[name]

		if !ok {
			return fmt.Errorf("graph.call: runner did not return output %q", name)
		}

		if err := execContext.Bind(ref, value); err != nil {
			return err
		}
	}

	return nil
}
