package integration

import (
	"context"
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/executor"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/scheduler"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
denoiserRunner is a deterministic stand-in for a diffusion denoiser
graph. Each call returns a velocity vector of fixed value 0.1 so the
runtime program can validate that the scheduler updates latents in
the expected direction without simulating a real diffusion model.
*/
type denoiserRunner struct{}

func (denoiserRunner) Call(
	execContext context.Context,
	module program.GraphModule,
	inputs map[string]any,
) (map[string]any, error) {
	latents, err := tensorValues(inputs["latents"])

	if err != nil {
		return nil, fmt.Errorf("denoiser: latents: %w", err)
	}

	velocity := make([]float64, len(latents))

	for index := range velocity {
		velocity[index] = 0.1
	}

	return map[string]any{"velocity": velocity}, nil
}

func tensorValues(value any) ([]float64, error) {
	switch typed := value.(type) {
	case *state.Tensor:
		return typed.Values(), nil
	case []float64:
		return typed, nil
	}

	return nil, fmt.Errorf("expected tensor or []float64, got %T", value)
}

func diffusionProgram(steps int) *program.Program {
	return &program.Program{
		Name: "diffusion",
		State: []program.StateDeclaration{
			{
				ID:   "latents",
				Type: "tensor",
				Config: map[string]any{
					"shape": []int{1, 4},
					"init":  "zeros",
				},
			},
		},
		Schedulers: []program.SchedulerDeclaration{
			{
				ID:   "scheduler",
				Type: "flow_match_euler_discrete",
				Config: map[string]any{
					"steps":               steps,
					"num_train_timesteps": 1000,
				},
			},
		},
		Graphs: map[string]program.GraphModule{
			"denoiser": {ID: "denoiser", Topology: "tiny.denoiser"},
		},
		Steps: []program.Step{
			{
				ID: "compute_timesteps",
				Op: "scheduler.timesteps",
				Inputs: map[string]program.ValueRef{
					"scheduler": {Namespace: program.NamespaceScheduler, Name: "scheduler"},
				},
				Outputs: map[string]program.ValueRef{
					"timesteps": {Namespace: program.NamespaceLocal, Name: "timesteps"},
				},
			},
			{
				ID: "denoise_loop",
				Op: "control.loop_each",
				Inputs: map[string]program.ValueRef{
					"source": {Namespace: program.NamespaceLocal, Name: "timesteps"},
				},
				Config: map[string]any{"as": "timestep"},
				Body: []program.Step{
					{
						ID: "denoise",
						Op: "graph.call",
						Inputs: map[string]program.ValueRef{
							"graph":    {Namespace: program.NamespaceGraph, Name: "denoiser"},
							"latents":  {Namespace: program.NamespaceState, Name: "latents"},
							"timestep": {Namespace: program.NamespaceLocal, Name: "timestep"},
						},
						Outputs: map[string]program.ValueRef{
							"velocity": {Namespace: program.NamespaceLocal, Name: "velocity"},
						},
					},
					{
						ID: "scheduler_step",
						Op: "scheduler.step",
						Inputs: map[string]program.ValueRef{
							"scheduler":  {Namespace: program.NamespaceScheduler, Name: "scheduler"},
							"step_index": {Namespace: program.NamespaceState, Name: "step_index_state"},
							"latents":    {Namespace: program.NamespaceState, Name: "latents"},
							"velocity":   {Namespace: program.NamespaceLocal, Name: "velocity"},
						},
						Outputs: map[string]program.ValueRef{
							"latents": {Namespace: program.NamespaceState, Name: "latents"},
						},
					},
					{
						ID:     "advance_index",
						Op:     "state.update",
						Config: map[string]any{"update": "increment"},
						Outputs: map[string]program.ValueRef{
							"target": {Namespace: program.NamespaceState, Name: "step_index_state"},
						},
					},
				},
			},
		},
	}
}

func TestDiffusionRuntimeEndToEnd(t *testing.T) {
	Convey("Given the canonical diffusion runtime program with 4 steps", t, func() {
		runtimeProgram := diffusionProgram(4)
		runtimeProgram.State = append(runtimeProgram.State, program.StateDeclaration{
			ID:     "step_index_state",
			Type:   "counter",
			Config: map[string]any{"initial": 0},
		})

		flowMatch := scheduler.NewFlowMatchEuler()

		exec, err := executor.New(executor.Options{
			Program:         runtimeProgram,
			GraphRunner:     denoiserRunner{},
			SchedulerRunner: flowMatch,
		})
		So(err, ShouldBeNil)

		Convey("Run should iterate the timesteps and update the latents tensor", func() {
			So(exec.Run(context.Background()), ShouldBeNil)

			tensorState := exec.States()["latents"].(*state.Tensor)
			values := tensorState.Values()

			Convey("Latents should no longer be zero after denoising", func() {
				So(len(values), ShouldEqual, 4)

				anyNonZero := false

				for _, value := range values {
					if value != 0 {
						anyNonZero = true
					}
				}

				So(anyNonZero, ShouldBeTrue)
			})

			Convey("Counter should report one iteration per scheduler timestep", func() {
				counter := exec.States()["step_index_state"].(*state.Counter)
				So(counter.Value(), ShouldEqual, 4)
			})
		})
	})
}
