package manifest

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

type passthroughOp struct{}

func (passthrough *passthroughOp) Forward(_ []int, data ...[]float64) []float64 {
	out := make([]float64, len(data[0]))
	copy(out, data[0])

	return out
}

func TestGraph_Execute(t *testing.T) {
	Convey("Given a single-node graph with a passthrough op", t, func() {
		graph := newGraph()

		graph.addNode(&Node{
			ID:  "copy",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		})

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
		})
	})
}

func TestGraph_rebuildEdgesFromNodes(t *testing.T) {
	Convey("Given a single-node graph wired only from graph inputs", t, func() {
		graph := newGraph()

		graph.addNode(&Node{
			ID:  "copy",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		})

		Convey("rebuildEdgesFromNodes", func() {
			Convey("It should produce no edges", func() {
				So(graph.rebuildEdgesFromNodes(), ShouldBeNil)
				So(graph.edges, ShouldBeEmpty)
			})
		})
	})

	Convey("Given a two-node chain sharing a binding", t, func() {
		graph := newGraph()

		graph.addNode(&Node{
			ID:  "upstream",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		})

		graph.addNode(&Node{
			ID:  "downstream",
			Op:  &passthroughOp{},
			In:  []string{"y"},
			Out: []string{"z"},
		})

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

		graph.addNode(&Node{
			ID:  "a",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		})

		graph.addNode(&Node{
			ID:  "b",
			Op:  &passthroughOp{},
			In:  []string{"x"},
			Out: []string{"y"},
		})

		Convey("rebuildEdgesFromNodes", func() {
			Convey("It should return an error", func() {
				So(graph.rebuildEdgesFromNodes(), ShouldNotBeNil)
			})
		})
	})
}

func BenchmarkGraph_Execute(b *testing.B) {
	graph := newGraph()

	graph.addNode(&Node{
		ID:  "copy",
		Op:  &passthroughOp{},
		In:  []string{"x"},
		Out: []string{"y"},
	})

	_ = graph.rebuildEdgesFromNodes()

	inputs := map[string][]float64{"x": make([]float64, 64)}
	shape := []int{64}

	b.ResetTimer()

	for repeat := 0; repeat < b.N; repeat++ {
		_, _ = graph.Execute(inputs, shape)
	}
}
