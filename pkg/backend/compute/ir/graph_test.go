package ir

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestGraph(t *testing.T) {
	Convey("Given a new Graph", t, func() {
		ctx := context.Background()
		graph := NewGraph(ctx)

		shape, _ := tensor.NewShape([]int{2, 2})
		nodeA := NewNode(ctx, "a", OpInput, shape)
		nodeB := NewNode(ctx, "b", OpMatmul, shape)
		nodeB.AddInput(nodeA)

		graph.AddNode(nodeA)
		graph.AddNode(nodeB)

		Convey("It should return all nodes", func() {
			nodes := graph.Nodes()
			So(len(nodes), ShouldEqual, 2)
		})

		Convey("It should correctly identify sink nodes", func() {
			sinks := graph.Sinks()
			So(len(sinks), ShouldEqual, 1)
			So(sinks[0].ID(), ShouldEqual, "b")
		})

		Convey("It should group nodes into concurrent topology layers", func() {
			nodeC := NewNode(ctx, "c", OpMatmul, shape)
			nodeC.AddInput(nodeA)
			graph.AddNode(nodeC)

			nodeD := NewNode(ctx, "d", OpAdd, shape)
			nodeD.AddInput(nodeB)
			nodeD.AddInput(nodeC)
			graph.AddNode(nodeD)

			layers := graph.TopologyLayers()

			// Layer 0: a
			// Layer 1: b, c
			// Layer 2: d
			So(len(layers), ShouldEqual, 3)
			So(len(layers[0]), ShouldEqual, 1)
			So(layers[0][0].ID(), ShouldEqual, "a")
			So(len(layers[1]), ShouldEqual, 2)
			So(len(layers[2]), ShouldEqual, 1)
			So(layers[2][0].ID(), ShouldEqual, "d")
		})
	})
}

func BenchmarkGraph(b *testing.B) {
	ctx := context.Background()
	shape, _ := tensor.NewShape([]int{2, 2})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graph := NewGraph(ctx)
		nodeA := NewNode(ctx, "a", OpInput, shape)
		nodeB := NewNode(ctx, "b", OpMatmul, shape)
		nodeB.AddInput(nodeA)
		graph.AddNode(nodeA)
		graph.AddNode(nodeB)
		graph.Sinks()
	}
}
