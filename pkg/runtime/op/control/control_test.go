package control

import (
	"errors"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

func TestLoopCount(t *testing.T) {
	Convey("Given a LoopCount op with count=3 in config", t, func() {
		stub := optest.NewStubContext()
		iterations := 0
		stub.BodyHandler = func(steps []program.Step) error {
			iterations++

			return nil
		}

		stub.StepRef = program.Step{
			ID:     "loop",
			Op:     "control.loop_count",
			Config: map[string]any{"count": 3},
			Body:   []program.Step{{ID: "x", Op: "test.noop"}},
		}

		Convey("Execute should run the body count times", func() {
			So(LoopCount{}.Execute(stub), ShouldBeNil)
			So(iterations, ShouldEqual, 3)
		})

		Convey("It should stop early when the body returns op.ErrBreak", func() {
			stub.BodyHandler = func(steps []program.Step) error {
				iterations++

				if iterations >= 2 {
					return op.ErrBreak
				}

				return nil
			}

			So(LoopCount{}.Execute(stub), ShouldBeNil)
			So(iterations, ShouldEqual, 2)
		})

		Convey("It should treat op.ErrContinue as a skip", func() {
			stub.BodyHandler = func(steps []program.Step) error {
				iterations++

				if iterations == 2 {
					return op.ErrContinue
				}

				return nil
			}

			So(LoopCount{}.Execute(stub), ShouldBeNil)
			So(iterations, ShouldEqual, 3)
		})

		Convey("It should resolve count from inputs when provided", func() {
			stub.Scope["max_new"] = 5
			stub.StepRef = program.Step{
				ID: "loop_from_input",
				Op: "control.loop_count",
				Inputs: map[string]program.ValueRef{
					"count": {Namespace: program.NamespaceLocal, Name: "max_new"},
				},
				Body: []program.Step{{ID: "x", Op: "test.noop"}},
			}

			So(LoopCount{}.Execute(stub), ShouldBeNil)
			So(iterations, ShouldEqual, 5)
		})
	})
}

func TestLoopEach(t *testing.T) {
	Convey("Given a LoopEach over a []float64 with as='timestep'", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["timesteps"] = []float64{1.0, 0.75, 0.5, 0.25}
		seen := []float64{}
		stub.BodyHandler = func(steps []program.Step) error {
			value, err := stub.Resolve(program.ValueRef{
				Namespace: program.NamespaceLocal,
				Name:      "timestep",
			})

			if err != nil {
				return err
			}

			seen = append(seen, value.(float64))

			return nil
		}

		stub.StepRef = program.Step{
			ID: "denoise_loop",
			Op: "control.loop_each",
			Inputs: map[string]program.ValueRef{
				"source": {Namespace: program.NamespaceLocal, Name: "timesteps"},
			},
			Config: map[string]any{"as": "timestep"},
			Body:   []program.Step{{ID: "step", Op: "test.noop"}},
		}

		Convey("It should iterate every element and bind the loop variable", func() {
			So(LoopEach{}.Execute(stub), ShouldBeNil)
			So(seen, ShouldResemble, []float64{1.0, 0.75, 0.5, 0.25})
		})
	})
}

func TestBreakIf(t *testing.T) {
	Convey("Given a BreakIf with a true condition local", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["stop"] = true
		stub.StepRef = program.Step{
			ID: "stop",
			Op: "control.break_if",
			Inputs: map[string]program.ValueRef{
				"condition": {Namespace: program.NamespaceLocal, Name: "stop"},
			},
		}

		Convey("It should return op.ErrBreak", func() {
			err := BreakIf{}.Execute(stub)
			So(errors.Is(err, op.ErrBreak), ShouldBeTrue)
		})

		Convey("It should return nil when the condition is false", func() {
			stub.Scope["stop"] = false
			err := BreakIf{}.Execute(stub)
			So(err, ShouldBeNil)
		})
	})
}
