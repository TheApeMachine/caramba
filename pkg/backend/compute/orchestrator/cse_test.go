package orchestrator

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestCSEOptimizer(t *testing.T) {
	Convey("Given a CSEOptimizer and an IR Graph", t, func() {
		optimizer := NewCSEOptimizer()
		graph := ir.NewGraph()

		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		nodeInput := ir.NewNode("in", ir.OpInput, shape)

		nodeBranchA := ir.NewNode("branch_a", ir.OpReLU, shape)
		nodeBranchA.AddInput(nodeInput)

		nodeBranchB := ir.NewNode("branch_b", ir.OpReLU, shape)
		nodeBranchB.AddInput(nodeInput)

		nodeSink := ir.NewNode("sink", ir.OpAdd, shape)
		nodeSink.AddInput(nodeBranchA)
		nodeSink.AddInput(nodeBranchB)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeBranchA)
		graph.AddNode(nodeBranchB)
		graph.AddNode(nodeSink)

		Convey("It should fold common subexpressions", func() {
			optimized, err := optimizer.Optimize(graph)
			So(err, ShouldBeNil)
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

			// Both inputs to the sink should now point to same node
			So(sink.Inputs()[0].ID(), ShouldEqual, sink.Inputs()[1].ID())
		})
	})
}

func BenchmarkCSEOptimizer(b *testing.B) {
	optimizer := NewCSEOptimizer()
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		graph := ir.NewGraph()
		nodeInput := ir.NewNode("in", ir.OpInput, shape)
		nodeBranchA := ir.NewNode("branch_a", ir.OpReLU, shape)
		nodeBranchA.AddInput(nodeInput)
		nodeBranchB := ir.NewNode("branch_b", ir.OpReLU, shape)
		nodeBranchB.AddInput(nodeInput)
		nodeSink := ir.NewNode("sink", ir.OpAdd, shape)
		nodeSink.AddInput(nodeBranchA)
		nodeSink.AddInput(nodeBranchB)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeBranchA)
		graph.AddNode(nodeBranchB)
		graph.AddNode(nodeSink)
		b.StartTimer()

		_, _ = optimizer.Optimize(graph)
	}
}
