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
		optimizer := NewDCEOptimizer(ctx)
		graph := ir.NewGraph(ctx)

		shape, _ := tensor.NewShape([]int{2, 2})
		nodeInput := ir.NewNode(ctx, "in", ir.OpInput, shape)

		nodeUsedPath := ir.NewNode(ctx, "used_path", ir.OpReLU, shape)
		nodeUsedPath.AddInput(nodeInput)

		nodeDeadPath := ir.NewNode(ctx, "dead_path", ir.OpGELU, shape)
		nodeDeadPath.AddInput(nodeInput)

		nodeSink := ir.NewNode(ctx, "sink", ir.OpAdd, shape)
		nodeSink.AddInput(nodeUsedPath)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeUsedPath)
		graph.AddNode(nodeDeadPath)
		graph.AddNode(nodeSink)

		Convey("It should remove dead nodes not connected to targets", func() {
			optimized := optimizer.Optimize(graph, []*ir.Node{nodeSink})
			nodes := optimized.Nodes()

			So(len(nodes), ShouldEqual, 3) // in, used_path, sink (dead_path removed)

			for _, n := range nodes {
				So(n.ID(), ShouldNotEqual, "dead_path")
			}
		})
	})
}

func BenchmarkDCEOptimizer(b *testing.B) {
	ctx := context.Background()
	optimizer := NewDCEOptimizer(ctx)
	shape, _ := tensor.NewShape([]int{2, 2})

	graph := ir.NewGraph(ctx)
	nodeInput := ir.NewNode(ctx, "in", ir.OpInput, shape)
	nodeSink := ir.NewNode(ctx, "sink", ir.OpAdd, shape)
	nodeSink.AddInput(nodeInput)

	nodeDead := ir.NewNode(ctx, "dead", ir.OpGELU, shape)
	nodeDead.AddInput(nodeInput)

	graph.AddNode(nodeInput)
	graph.AddNode(nodeSink)
	graph.AddNode(nodeDead)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.Optimize(graph, nil)
	}
}
