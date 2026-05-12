package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestFusionOptimizer(t *testing.T) {
	Convey("Given a FusionOptimizer and an IR Graph", t, func() {
		ctx := context.Background()
		optimizer := NewFusionOptimizer(ctx)
		graph := ir.NewGraph(ctx)

		shape, _ := tensor.NewShape([]int{2, 2})
		nodeInput := ir.NewNode(ctx, "in", ir.OpInput, shape)
		nodeMatmul := ir.NewNode(ctx, "matmul", ir.OpMatmul, shape)
		nodeReLU := ir.NewNode(ctx, "relu", ir.OpReLU, shape)

		nodeMatmul.AddInput(nodeInput)
		nodeReLU.AddInput(nodeMatmul)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeMatmul)
		graph.AddNode(nodeReLU)

		Convey("It should fuse Matmul and ReLU into a single node", func() {
			optimized := optimizer.Optimize(graph)
			nodes := optimized.Nodes()

			So(len(nodes), ShouldEqual, 2)

			fusedFound := false
			for _, n := range nodes {
				if n.OpType() == ir.OpFused {
					fusedFound = true
					So(n.ID(), ShouldEqual, "matmul_fused_relu")
					So(n.Metadata()["base_op"], ShouldEqual, "Matmul")
					So(n.Metadata()["activation"], ShouldEqual, "ReLU")
					So(len(n.Inputs()), ShouldEqual, 1)
					So(n.Inputs()[0].ID(), ShouldEqual, "in")
				}
			}
			So(fusedFound, ShouldBeTrue)
		})
	})
}

func BenchmarkFusionOptimizer(b *testing.B) {
	ctx := context.Background()
	optimizer := NewFusionOptimizer(ctx)
	shape, _ := tensor.NewShape([]int{2, 2})

	// Pre-build graph
	graph := ir.NewGraph(ctx)
	nodeInput := ir.NewNode(ctx, "in", ir.OpInput, shape)
	nodeMatmul := ir.NewNode(ctx, "matmul", ir.OpMatmul, shape)
	nodeReLU := ir.NewNode(ctx, "relu", ir.OpReLU, shape)

	nodeMatmul.AddInput(nodeInput)
	nodeReLU.AddInput(nodeMatmul)

	graph.AddNode(nodeInput)
	graph.AddNode(nodeMatmul)
	graph.AddNode(nodeReLU)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.Optimize(graph)
	}
}
