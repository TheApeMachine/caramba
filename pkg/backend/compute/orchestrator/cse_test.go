package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestCSEOptimizer(t *testing.T) {
	Convey("Given a CSEOptimizer and an IR Graph", t, func() {
		ctx := context.Background()
		optimizer := NewCSEOptimizer(ctx)
		graph := ir.NewGraph(ctx)

		shape, _ := tensor.NewShape([]int{2, 2})
		nodeInput := ir.NewNode(ctx, "in", ir.OpInput, shape)

		nodeBranchA := ir.NewNode(ctx, "branch_a", ir.OpReLU, shape)
		nodeBranchA.AddInput(nodeInput)

		nodeBranchB := ir.NewNode(ctx, "branch_b", ir.OpReLU, shape)
		nodeBranchB.AddInput(nodeInput)

		nodeSink := ir.NewNode(ctx, "sink", ir.OpAdd, shape)
		nodeSink.AddInput(nodeBranchA)
		nodeSink.AddInput(nodeBranchB)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeBranchA)
		graph.AddNode(nodeBranchB)
		graph.AddNode(nodeSink)

		Convey("It should fold common subexpressions", func() {
			optimized := optimizer.Optimize(graph)
			nodes := optimized.Nodes()

			So(len(nodes), ShouldEqual, 3) // in, branch_a, sink (branch_b folded)

			// Find the sink node
			var sink *ir.Node
			for _, n := range nodes {
				if n.ID() == "sink" {
					sink = n
				}
			}

			So(sink, ShouldNotBeNil)
			So(len(sink.Inputs()), ShouldEqual, 2)

			// Both inputs to the sink should now point to branch_a
			So(sink.Inputs()[0].ID(), ShouldEqual, "branch_a")
			So(sink.Inputs()[1].ID(), ShouldEqual, "branch_a")
		})
	})
}

func BenchmarkCSEOptimizer(b *testing.B) {
	ctx := context.Background()
	optimizer := NewCSEOptimizer(ctx)
	shape, _ := tensor.NewShape([]int{2, 2})

	graph := ir.NewGraph(ctx)
	nodeInput := ir.NewNode(ctx, "in", ir.OpInput, shape)
	nodeBranchA := ir.NewNode(ctx, "branch_a", ir.OpReLU, shape)
	nodeBranchA.AddInput(nodeInput)
	nodeBranchB := ir.NewNode(ctx, "branch_b", ir.OpReLU, shape)
	nodeBranchB.AddInput(nodeInput)

	graph.AddNode(nodeInput)
	graph.AddNode(nodeBranchA)
	graph.AddNode(nodeBranchB)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.Optimize(graph)
	}
}
