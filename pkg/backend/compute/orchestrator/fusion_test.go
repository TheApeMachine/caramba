package orchestrator

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestFusionOptimizer(t *testing.T) {
	Convey("Given a FusionOptimizer and an IR Graph", t, func() {
		optimizer := NewFusionOptimizer()
		graph := ir.NewGraph()

		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		nodeInput := ir.NewNode("in", ir.OpInput, shape)
		nodeMatmul := ir.NewNode("matmul", ir.OpMatmul, shape)
		nodeMatmul.SetMetadata("custom_meta", "value")

		nodeReLU := ir.NewNode("relu", ir.OpReLU, shape)
		nodeReLU.SetMetadata("act_meta", "val2")

		nodeMatmul.AddInput(nodeInput)
		nodeReLU.AddInput(nodeMatmul)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeMatmul)
		graph.AddNode(nodeReLU)

		Convey("It should fuse Matmul and ReLU into a single node", func() {
			optimized, err := optimizer.Optimize(graph)
			So(err, ShouldBeNil)
			nodes := optimized.Nodes()

			So(len(nodes), ShouldEqual, 2)

			fusedFound := false
			for _, n := range nodes {
				if n.OpType() == ir.OpFused {
					fusedFound = true
					So(n.ID(), ShouldEqual, "matmul_fused_relu")
					So(n.Metadata()["base_op"], ShouldEqual, "Matmul")
					So(n.Metadata()["activation"], ShouldEqual, "ReLU")
					So(n.Metadata()["custom_meta"], ShouldEqual, "value")
					So(n.Metadata()["act_meta"], ShouldEqual, "val2")
					So(len(n.Inputs()), ShouldEqual, 1)
					So(n.Inputs()[0].ID(), ShouldEqual, "in")
				}
			}
			So(fusedFound, ShouldBeTrue)
		})

		Convey("It should return original graph topology if no fusion targets exist", func() {
			graphNoFuse := ir.NewGraph()
			n1 := ir.NewNode("n1", ir.OpInput, shape)
			n2 := ir.NewNode("n2", ir.OpAdd, shape)
			n2.AddInput(n1)
			graphNoFuse.AddNode(n1)
			graphNoFuse.AddNode(n2)

			optimized, err := optimizer.Optimize(graphNoFuse)
			So(err, ShouldBeNil)
			So(len(optimized.Nodes()), ShouldEqual, 2)
		})

		Convey("It should be idempotent", func() {
			optimized1, _ := optimizer.Optimize(graph)
			optimized2, _ := optimizer.Optimize(optimized1)

			So(len(optimized1.Nodes()), ShouldEqual, len(optimized2.Nodes()))
		})

		Convey("It should handle nil graph", func() {
			_, err := optimizer.Optimize(nil)
			So(err, ShouldNotBeNil)
		})
	})
}

func BenchmarkFusionOptimizer(b *testing.B) {
	optimizer := NewFusionOptimizer()
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// FusionOptimizer is pure and doesn't mutate the input graph,
		// but to be perfectly strict we rebuild the small test graph
		b.StopTimer()
		graph := ir.NewGraph()
		nodeInput := ir.NewNode("in", ir.OpInput, shape)
		nodeMatmul := ir.NewNode("matmul", ir.OpMatmul, shape)
		nodeReLU := ir.NewNode("relu", ir.OpReLU, shape)

		nodeMatmul.AddInput(nodeInput)
		nodeReLU.AddInput(nodeMatmul)

		graph.AddNode(nodeInput)
		graph.AddNode(nodeMatmul)
		graph.AddNode(nodeReLU)
		b.StartTimer()

		_, _ = optimizer.Optimize(graph)
	}
}
