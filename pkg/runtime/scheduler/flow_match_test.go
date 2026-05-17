package scheduler

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

func TestFlowMatchEulerTimesteps(t *testing.T) {
	Convey("Given a FlowMatchEuler runner and a 4-step declaration", t, func() {
		runner := NewFlowMatchEuler()
		declaration := program.SchedulerDeclaration{
			ID:   "sched",
			Type: "flow_match_euler_discrete",
			Config: map[string]any{
				"steps":               4,
				"num_train_timesteps": 1000,
			},
		}

		Convey("Timesteps should return 4 monotonically decreasing values", func() {
			timesteps, err := runner.Timesteps(context.Background(), declaration)
			So(err, ShouldBeNil)
			So(len(timesteps), ShouldEqual, 4)

			for index := 1; index < len(timesteps); index++ {
				So(timesteps[index], ShouldBeLessThan, timesteps[index-1])
			}
		})
	})
}

func TestFlowMatchEulerStep(t *testing.T) {
	Convey("Given a FlowMatchEuler runner and a 4-step declaration", t, func() {
		runner := NewFlowMatchEuler()
		declaration := program.SchedulerDeclaration{
			ID:   "sched",
			Type: "flow_match_euler_discrete",
			Config: map[string]any{
				"steps":               4,
				"num_train_timesteps": 1000,
			},
		}

		latents := []float64{1, 2, 3, 4}
		velocity := []float64{0.1, 0.1, 0.1, 0.1}

		Convey("Step 0 should update latents by sigma delta * velocity", func() {
			updated, err := runner.Step(context.Background(), declaration, 0, latents, velocity)
			So(err, ShouldBeNil)
			So(len(updated), ShouldEqual, 4)

			for index := range latents {
				So(updated[index], ShouldNotEqual, latents[index])
			}
		})

		Convey("Step beyond the final index should error", func() {
			_, err := runner.Step(context.Background(), declaration, 5, latents, velocity)
			So(err, ShouldNotBeNil)
		})

		Convey("Mismatched lengths should error", func() {
			_, err := runner.Step(context.Background(), declaration, 0, latents, []float64{0.1})
			So(err, ShouldNotBeNil)
		})
	})
}
