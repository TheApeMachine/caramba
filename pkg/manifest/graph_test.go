package manifest

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type passthroughOp struct{}

func (passthrough *passthroughOp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("passthrough"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 || stateDict.Inputs[0] == nil {
		return nil, fmt.Errorf("passthrough: input[0] is required")
	}

	if len(stateDict.Out) != len(stateDict.Inputs[0]) {
		stateDict.Out = make([]float64, len(stateDict.Inputs[0]))
	}

	copy(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}

func TestGraph_Execute(t *testing.T) {
	Convey("Given a single-node graph with a passthrough op", t, func() {
		graph := newGraph()

		So(graph.addNode(&Node{
			ID:  "copy",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		}), ShouldBeNil)
		graph.externalInputs["x"] = true

		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		Convey("Execute", func() {
			Convey("It should propagate input to the named output port", func() {
				state, err := graph.Execute(map[string][]float64{"x": {1, 2, 3}}, []int{3})
				So(err, ShouldBeNil)
				So(state["y"], ShouldResemble, []float64{1, 2, 3})
			})

			Convey("It should reject missing input bindings", func() {
				_, err := graph.Execute(map[string][]float64{}, []int{3})
				So(err, ShouldNotBeNil)
			})

			Convey("It should reject nil external input slices", func() {
				_, err := graph.Execute(map[string][]float64{"x": nil}, []int{3})
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "input[0] is required")
			})
		})
	})
}

func TestGraph_rebuildEdgesFromNodes(t *testing.T) {
	Convey("Given a single-node graph wired only from graph inputs", t, func() {
		graph := newGraph()

		So(graph.addNode(&Node{
			ID:  "copy",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		}), ShouldBeNil)
		graph.externalInputs["x"] = true

		Convey("rebuildEdgesFromNodes", func() {
			Convey("It should produce no edges", func() {
				So(graph.rebuildEdgesFromNodes(), ShouldBeNil)
				So(graph.edges, ShouldBeEmpty)
			})
		})
	})

	Convey("Given a two-node chain sharing a binding", t, func() {
		graph := newGraph()

		So(graph.addNode(&Node{
			ID:  "upstream",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		}), ShouldBeNil)

		So(graph.addNode(&Node{
			ID:  "downstream",
			Op:  &passthroughOp{},
			In:  []string{"y"},
			Out: []string{"z"},
		}), ShouldBeNil)
		graph.externalInputs["x"] = true

		Convey("rebuildEdgesFromNodes", func() {
			Convey("It should add one edge from producer to consumer", func() {
				So(graph.rebuildEdgesFromNodes(), ShouldBeNil)
				So(graph.edges, ShouldHaveLength, 1)
				So(graph.edges[0].From, ShouldEqual, "upstream")
				So(graph.edges[0].To, ShouldEqual, "downstream")
				So(graph.edges[0].FromPort, ShouldEqual, "y")
				So(graph.edges[0].ToPort, ShouldEqual, "y")
			})
		})
	})

	Convey("Given two nodes that publish the same binding", t, func() {
		graph := newGraph()

		So(graph.addNode(&Node{
			ID:  "a",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		}), ShouldBeNil)

		So(graph.addNode(&Node{
			ID:  "b",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		}), ShouldBeNil)
		graph.externalInputs["x"] = true

		Convey("rebuildEdgesFromNodes", func() {
			Convey("It should return an error", func() {
				So(graph.rebuildEdgesFromNodes(), ShouldNotBeNil)
			})
		})
	})

	Convey("Given duplicate node IDs", t, func() {
		graph := newGraph()
		So(graph.addNode(&Node{ID: "dup", Op: &passthroughOp{}, Out: []string{"a"}}), ShouldBeNil)

		Convey("addNode should reject the duplicate", func() {
			err := graph.addNode(&Node{ID: "dup", Op: &passthroughOp{}, Out: []string{"b"}})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "duplicate node id")
		})
	})

	Convey("Given a missing producer", t, func() {
		graph := newGraph()
		So(graph.addNode(&Node{ID: "consumer", Op: &passthroughOp{}, In: []string{"missing"}, Out: []string{"y"}}), ShouldBeNil)

		Convey("rebuildEdgesFromNodes should reject the unresolved binding", func() {
			err := graph.rebuildEdgesFromNodes()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "has no producer or declared external input")
		})
	})

	Convey("Given a multi-output node", t, func() {
		graph := newGraph()

		Convey("addNode should reject unsupported multi-output declarations", func() {
			err := graph.addNode(&Node{ID: "multi", Op: &passthroughOp{}, Out: []string{"a", "b"}})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "multi-output operations are not supported")
		})
	})
}

func TestGraph_ExecuteTopology(t *testing.T) {
	Convey("Given nodes added out of topological order", t, func() {
		graph := newGraph()
		graph.externalInputs["x"] = true

		So(graph.addNode(&Node{ID: "downstream", Op: &passthroughOp{}, In: []string{"mid"}, Out: []string{"y"}}), ShouldBeNil)
		So(graph.addNode(&Node{ID: "upstream", Op: &passthroughOp{}, In: []string{"x"}, Out: []string{"mid"}}), ShouldBeNil)
		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		Convey("Execute should run in dependency order", func() {
			state, err := graph.Execute(map[string][]float64{"x": {1, 2, 3}}, []int{3})

			So(err, ShouldBeNil)
			So(state["y"], ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func BenchmarkGraph_Execute(b *testing.B) {
	graph := newGraph()

	_ = graph.addNode(&Node{
		ID:  "copy",
		Op:  &passthroughOp{},
		In:  []string{"x"},
		Out: []string{"y"},
	})
	graph.externalInputs["x"] = true

	_ = graph.rebuildEdgesFromNodes()

	inputs := map[string][]float64{"x": make([]float64, 64)}
	shape := []int{64}

	b.ResetTimer()

	for b.Loop() {
		_, _ = graph.Execute(inputs, shape)
	}
}
