package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestDCEOptimizer(t *testing.T) {
	Convey("Given a DCEOptimizer and an IR Graph", t, func() {
		ctx := context.Background()
		optimizer := NewDCEOptimizer()
		graph := ir.NewGraph()

		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		nodeInput := ir.NewNode("in", ir.OpInput, shape)

		nodeUsedPath := ir.NewNode("used_path", ir.OpReLU, shape)
		nodeUsedPath.AddInput(nodeInput)

		nodeDeadPath := ir.NewNode("dead_path", ir.OpGELU, shape)
		nodeDeadPath.AddInput(nodeInput)

		nodeSink := ir.NewNode("sink", ir.OpAdd, shape)
		nodeSink.AddInput(nodeUsedPath)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeUsedPath)
		graph.AddNode(nodeDeadPath)
		graph.AddNode(nodeSink)

		Convey("It should remove dead nodes not connected to targets", func() {
			optimized, err := optimizer.Optimize(ctx, graph, []*ir.Node{nodeSink})
			So(err, ShouldBeNil)
			nodes := optimized.Nodes()

			So(len(nodes), ShouldEqual, 3) // in, used_path, sink (dead_path removed)

			var ids []string
			for _, n := range nodes {
				ids = append(ids, n.ID())
			}
			So(ids, ShouldContain, "in")
			So(ids, ShouldContain, "used_path")
			So(ids, ShouldContain, "sink")
			So(ids, ShouldNotContain, "dead_path")

			// Verify edges are intact
			for _, n := range nodes {
				if n.ID() == "sink" {
					So(n.Inputs()[0].ID(), ShouldEqual, "used_path")
				}
				if n.ID() == "used_path" {
					So(n.Inputs()[0].ID(), ShouldEqual, "in")
				}
			}
		})
	})
}

func BenchmarkDCEOptimizer(b *testing.B) {
	ctx := context.Background()
	optimizer := NewDCEOptimizer()
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		graph := ir.NewGraph()
		nodeInput := ir.NewNode("in", ir.OpInput, shape)
		nodeSink := ir.NewNode("sink", ir.OpAdd, shape)
		nodeSink.AddInput(nodeInput)

		nodeDead := ir.NewNode("dead", ir.OpGELU, shape)
		nodeDead.AddInput(nodeInput)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeSink)
		graph.AddNode(nodeDead)
		targets := []*ir.Node{nodeSink}
		b.StartTimer()

		_, _ = optimizer.Optimize(ctx, graph, targets)
	}
}
