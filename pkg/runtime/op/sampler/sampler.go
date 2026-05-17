package sampler

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
NextToken consumes a logits vector and the current decode history,
asks the executor's SamplerRunner to produce the next token, and
writes the result to Outputs["token"]. When the sampler reports its
stop condition was matched the op writes that flag to a sibling
output named "stopped" if one is declared.
*/
type NextToken struct{}

func (NextToken) Execute(execContext op.Context) error {
	step := execContext.Step()

	runner := execContext.SamplerRunner()

	if runner == nil {
		return fmt.Errorf("sampler.next_token: no SamplerRunner is wired up")
	}

	samplerRef, ok := step.Inputs["sampler"]

	if !ok || samplerRef.Namespace != program.NamespaceSampler {
		return fmt.Errorf("sampler.next_token: inputs.sampler must reference the sampler namespace")
	}

	declaration, err := execContext.Sampler(samplerRef.Name)

	if err != nil {
		return err
	}

	logits, err := resolveLogits(execContext, step.Inputs)

	if err != nil {
		return err
	}

	history, err := resolveHistory(execContext, step.Inputs)

	if err != nil {
		return err
	}

	token, stopped, err := runner.Next(execContext.Context(), declaration, logits, history)

	if err != nil {
		return fmt.Errorf("sampler.next_token: %w", err)
	}

	tokenOutput, ok := step.Outputs["token"]

	if !ok {
		return fmt.Errorf("sampler.next_token: missing output 'token'")
	}

	if err := execContext.Bind(tokenOutput, token); err != nil {
		return err
	}

	if stoppedOutput, ok := step.Outputs["stopped"]; ok {
		return execContext.Bind(stoppedOutput, stopped)
	}

	return nil
}

func init() {
	op.Default.MustRegister("sampler.next_token", NextToken{})
}

func resolveLogits(
	execContext op.Context, inputs map[string]program.ValueRef,
) ([]float64, error) {
	ref, ok := inputs["logits"]

	if !ok {
		return nil, fmt.Errorf("sampler.next_token: missing input 'logits'")
	}

	raw, err := execContext.Resolve(ref)

	if err != nil {
		return nil, err
	}

	switch typed := raw.(type) {
	case []float64:
		return typed, nil
	case []float32:
		out := make([]float64, len(typed))

		for index, value := range typed {
			out[index] = float64(value)
		}

		return out, nil
	}

	return nil, fmt.Errorf("sampler.next_token: logits must be []float64, got %T", raw)
}

func resolveHistory(
	execContext op.Context, inputs map[string]program.ValueRef,
) ([]int, error) {
	ref, ok := inputs["history"]

	if !ok {
		return nil, nil
	}

	raw, err := execContext.Resolve(ref)

	if err != nil {
		return nil, err
	}

	switch typed := raw.(type) {
	case []int:
		return typed, nil
	case *state.TokenBuffer:
		return typed.Tokens(), nil
	case state.State:
		buffer, ok := typed.(*state.TokenBuffer)

		if !ok {
			return nil, fmt.Errorf(
				"sampler.next_token: history state must be token_buffer, got %q",
				typed.Type(),
			)
		}

		return buffer.Tokens(), nil
	}

	return nil, fmt.Errorf("sampler.next_token: history must be []int or token_buffer, got %T", raw)
}
