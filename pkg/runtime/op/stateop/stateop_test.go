package stateop

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

func TestStateReset(t *testing.T) {
	Convey("Given a Reset op targeting a token_buffer", t, func() {
		stub := optest.NewStubContext()
		buffer, _ := state.Default.Build("token_buffer", "history", nil)
		stub.States["history"] = buffer
		buffer.(*state.TokenBuffer).Extend([]int{1, 2, 3})

		stub.StepRef = program.Step{
			ID: "reset",
			Op: "state.reset",
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceState, Name: "history"},
			},
		}

		Convey("Execute should clear the buffer", func() {
			So(ResetState{}.Execute(stub), ShouldBeNil)
			So(buffer.(*state.TokenBuffer).Length(), ShouldEqual, 0)
		})
	})
}

func TestStateAppend(t *testing.T) {
	Convey("Given an Append op writing an int to a token_buffer", t, func() {
		stub := optest.NewStubContext()
		buffer, _ := state.Default.Build("token_buffer", "history", nil)
		stub.States["history"] = buffer
		stub.Scope["token"] = 42

		stub.StepRef = program.Step{
			ID: "append",
			Op: "state.append",
			Inputs: map[string]program.ValueRef{
				"value": {Namespace: program.NamespaceLocal, Name: "token"},
			},
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceState, Name: "history"},
			},
		}

		Convey("Execute should append the token", func() {
			So(Append{}.Execute(stub), ShouldBeNil)
			So(buffer.(*state.TokenBuffer).Tokens(), ShouldResemble, []int{42})
		})
	})
}

func TestCounterIncrement(t *testing.T) {
	Convey("Given an Update increment on a counter", t, func() {
		stub := optest.NewStubContext()
		counter, _ := state.Default.Build("counter", "position", map[string]any{"initial": 0})
		stub.States["position"] = counter

		stub.StepRef = program.Step{
			ID:     "advance",
			Op:     "state.update",
			Config: map[string]any{"update": "increment"},
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceState, Name: "position"},
			},
		}

		Convey("Execute should increment by 1 when no delta is configured", func() {
			So(Update{}.Execute(stub), ShouldBeNil)
			So(counter.(*state.Counter).Value(), ShouldEqual, 1)
		})

		Convey("Execute should respect config.delta", func() {
			stub.StepRef.Config["delta"] = 3
			So(Update{}.Execute(stub), ShouldBeNil)
			So(counter.(*state.Counter).Value(), ShouldEqual, 3)
		})

		Convey("Execute should prefer inputs.delta when it is supplied", func() {
			stub.Scope["delta"] = 4
			stub.StepRef.Config["delta"] = 2
			stub.StepRef.Inputs = map[string]program.ValueRef{
				"delta": {Namespace: program.NamespaceLocal, Name: "delta"},
			}

			So(Update{}.Execute(stub), ShouldBeNil)
			So(counter.(*state.Counter).Value(), ShouldEqual, 4)
		})
	})
}

func BenchmarkUpdate_CounterDeltaInput(benchmark *testing.B) {
	stub := optest.NewStubContext()
	counter, err := state.Default.Build("counter", "position", map[string]any{"initial": 0})

	if err != nil {
		benchmark.Fatal(err)
	}

	stub.States["position"] = counter
	stub.Scope["delta"] = 4
	stub.StepRef = program.Step{
		ID:     "advance",
		Op:     "state.update",
		Config: map[string]any{"update": "increment"},
		Inputs: map[string]program.ValueRef{
			"delta": {Namespace: program.NamespaceLocal, Name: "delta"},
		},
		Outputs: map[string]program.ValueRef{
			"target": {Namespace: program.NamespaceState, Name: "position"},
		},
	}

	for benchmark.Loop() {
		if err := (Update{}).Execute(stub); err != nil {
			benchmark.Fatal(err)
		}
	}
}
