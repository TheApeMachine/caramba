package scheduler

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
Timesteps asks the executor's SchedulerRunner to compute the timestep
schedule for the declared scheduler and writes the resulting
[]float64 to Outputs["timesteps"].
*/
type Timesteps struct{}

func (Timesteps) Execute(execContext op.Context) error {
	step := execContext.Step()

	runner := execContext.SchedulerRunner()

	if runner == nil {
		return fmt.Errorf("scheduler.timesteps: no SchedulerRunner is wired up")
	}

	declaration, err := schedulerFromInputs(execContext, step.Inputs)

	if err != nil {
		return err
	}

	output, ok := step.Outputs["timesteps"]

	if !ok {
		return fmt.Errorf("scheduler.timesteps: missing output 'timesteps'")
	}

	values, err := runner.Timesteps(execContext.Context(), declaration)

	if err != nil {
		return fmt.Errorf("scheduler.timesteps: %w", err)
	}

	return execContext.Bind(output, values)
}

/*
Step advances the latents one denoising step. It expects:
  - Inputs["scheduler"]: scheduler declaration
  - Inputs["step_index"]: int
  - Inputs["latents"]: tensor state or []float64
  - Inputs["velocity"]: []float64 (model output)
Outputs["latents"] receives the updated latent values. When the
target is a tensor state object the op writes back through Set so
the residency layer remains responsible.
*/
type Step struct{}

func (Step) Execute(execContext op.Context) error {
	step := execContext.Step()

	runner := execContext.SchedulerRunner()

	if runner == nil {
		return fmt.Errorf("scheduler.step: no SchedulerRunner is wired up")
	}

	declaration, err := schedulerFromInputs(execContext, step.Inputs)

	if err != nil {
		return err
	}

	stepIndexRef, ok := step.Inputs["step_index"]

	if !ok {
		return fmt.Errorf("scheduler.step: missing input 'step_index'")
	}

	stepIndexValue, err := execContext.Resolve(stepIndexRef)

	if err != nil {
		return err
	}

	stepIndex, err := asInt(stepIndexValue)

	if err != nil {
		return fmt.Errorf("scheduler.step: step_index: %w", err)
	}

	latentsRef, ok := step.Inputs["latents"]

	if !ok {
		return fmt.Errorf("scheduler.step: missing input 'latents'")
	}

	latents, latentsShape, latentsTensor, err := resolveTensorValues(execContext, latentsRef)

	if err != nil {
		return err
	}

	velocityRef, ok := step.Inputs["velocity"]

	if !ok {
		return fmt.Errorf("scheduler.step: missing input 'velocity'")
	}

	velocity, _, _, err := resolveTensorValues(execContext, velocityRef)

	if err != nil {
		return err
	}

	updated, err := runner.Step(execContext.Context(), declaration, stepIndex, latents, velocity)

	if err != nil {
		return fmt.Errorf("scheduler.step: %w", err)
	}

	output, ok := step.Outputs["latents"]

	if !ok {
		return fmt.Errorf("scheduler.step: missing output 'latents'")
	}

	if latentsTensor != nil && output.Namespace == program.NamespaceState && output.Name == latentsRef.Name {
		return latentsTensor.Set(latentsShape, updated)
	}

	return execContext.Bind(output, updated)
}

func init() {
	op.Default.MustRegister("scheduler.timesteps", Timesteps{})
	op.Default.MustRegister("scheduler.step", Step{})
}

func schedulerFromInputs(
	execContext op.Context, inputs map[string]program.ValueRef,
) (program.SchedulerDeclaration, error) {
	ref, ok := inputs["scheduler"]

	if !ok || ref.Namespace != program.NamespaceScheduler {
		return program.SchedulerDeclaration{}, fmt.Errorf(
			"scheduler: inputs.scheduler must reference the scheduler namespace",
		)
	}

	return execContext.Scheduler(ref.Name)
}

func resolveTensorValues(
	execContext op.Context, ref program.ValueRef,
) ([]float64, []int, *state.Tensor, error) {
	raw, err := execContext.Resolve(ref)

	if err != nil {
		return nil, nil, nil, err
	}

	switch typed := raw.(type) {
	case *state.Tensor:
		return typed.Values(), typed.Shape(), typed, nil
	case []float64:
		return typed, nil, nil, nil
	}

	return nil, nil, nil, fmt.Errorf("expected tensor or []float64, got %T", raw)
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
	case *state.Counter:
		return typed.Value(), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}
