package value

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

func TestAssign(t *testing.T) {
	Convey("Given a value.assign step", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["source"] = 42
		stub.StepRef = program.Step{
			ID: "carry",
			Op: "value.assign",
			Inputs: map[string]program.ValueRef{
				"value": {Namespace: program.NamespaceLocal, Name: "source"},
			},
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceLocal, Name: "carry"},
			},
		}

		Convey("Execute should copy the value to the target", func() {
			So(Assign{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["carry"], ShouldEqual, 42)
		})
	})
}

func TestAppend(t *testing.T) {
	Convey("Given an Append onto a []int", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["tokens"] = []int{1, 2, 3}
		stub.Scope["next"] = 4
		stub.StepRef = program.Step{
			ID: "append",
			Op: "value.append",
			Inputs: map[string]program.ValueRef{
				"element": {Namespace: program.NamespaceLocal, Name: "next"},
			},
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceLocal, Name: "tokens"},
			},
		}

		Convey("Execute should extend the slice in place", func() {
			So(Append{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["tokens"], ShouldResemble, []int{1, 2, 3, 4})
		})
	})
}

func TestSlice(t *testing.T) {
	Convey("Given a Slice op extracting [1:3] from a []int", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["source"] = []int{10, 20, 30, 40}
		stub.StepRef = program.Step{
			ID: "slice",
			Op: "value.slice",
			Inputs: map[string]program.ValueRef{
				"source": {Namespace: program.NamespaceLocal, Name: "source"},
			},
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceLocal, Name: "out"},
			},
			Config: map[string]any{"start": 1, "end": 3},
		}

		Convey("Execute should bind the sub-slice", func() {
			So(Slice{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["out"], ShouldResemble, []int{20, 30})
		})
	})
}

func TestClear(t *testing.T) {
	Convey("Given a Clear on a []int target", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["history"] = []int{1, 2, 3}
		stub.StepRef = program.Step{
			ID: "clear",
			Op: "value.clear",
			Outputs: map[string]program.ValueRef{
				"target": {Namespace: program.NamespaceLocal, Name: "history"},
			},
		}

		Convey("Execute should reset the slice to empty", func() {
			So(Clear{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["history"], ShouldResemble, []int{})
		})
	})
}

func TestLength(t *testing.T) {
	Convey("Given a Length op measuring input tokens", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["tokens"] = []int{10, 20, 30}
		stub.StepRef = program.Step{
			ID: "length",
			Op: "value.length",
			Inputs: map[string]program.ValueRef{
				"value": {Namespace: program.NamespaceLocal, Name: "tokens"},
			},
			Outputs: map[string]program.ValueRef{
				"length": {Namespace: program.NamespaceLocal, Name: "token_count"},
			},
		}

		Convey("Execute should bind the element count", func() {
			So(Length{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["token_count"], ShouldEqual, 3)
		})
	})
}

func TestPositions(t *testing.T) {
	Convey("Given a Positions op with a counter start", t, func() {
		stub := optest.NewStubContext()
		counter, err := state.Default.Build("counter", "position", map[string]any{"initial": 7})
		So(err, ShouldBeNil)
		stub.States["position"] = counter
		stub.Scope["tokens"] = []int{101, 102, 103}
		stub.StepRef = program.Step{
			ID: "positions",
			Op: "value.positions",
			Inputs: map[string]program.ValueRef{
				"start":  {Namespace: program.NamespaceState, Name: "position"},
				"tokens": {Namespace: program.NamespaceLocal, Name: "tokens"},
			},
			Outputs: map[string]program.ValueRef{
				"positions": {Namespace: program.NamespaceLocal, Name: "position_ids"},
			},
		}

		Convey("Execute should bind contiguous position ids", func() {
			So(Positions{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["position_ids"], ShouldResemble, []int{7, 8, 9})
		})
	})
}

func BenchmarkLength_Execute(benchmark *testing.B) {
	stub := optest.NewStubContext()
	stub.Scope["tokens"] = []int{10, 20, 30, 40}
	stub.StepRef = program.Step{
		ID: "length",
		Op: "value.length",
		Inputs: map[string]program.ValueRef{
			"value": {Namespace: program.NamespaceLocal, Name: "tokens"},
		},
		Outputs: map[string]program.ValueRef{
			"length": {Namespace: program.NamespaceLocal, Name: "token_count"},
		},
	}

	for benchmark.Loop() {
		if err := (Length{}).Execute(stub); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkPositions_Execute(benchmark *testing.B) {
	stub := optest.NewStubContext()
	counter, err := state.Default.Build("counter", "position", map[string]any{"initial": 7})

	if err != nil {
		benchmark.Fatal(err)
	}

	stub.States["position"] = counter
	stub.Scope["tokens"] = []int{101, 102, 103, 104}
	stub.StepRef = program.Step{
		ID: "positions",
		Op: "value.positions",
		Inputs: map[string]program.ValueRef{
			"start":  {Namespace: program.NamespaceState, Name: "position"},
			"tokens": {Namespace: program.NamespaceLocal, Name: "tokens"},
		},
		Outputs: map[string]program.ValueRef{
			"positions": {Namespace: program.NamespaceLocal, Name: "position_ids"},
		},
	}

	for benchmark.Loop() {
		if err := (Positions{}).Execute(stub); err != nil {
			benchmark.Fatal(err)
		}
	}
}
