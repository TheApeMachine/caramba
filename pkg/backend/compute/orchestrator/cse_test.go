package orchestrator

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
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

		Convey("It should preserve semantics on surviving representatives", func() {
			nodeBranchA.SetOperationID("custom.relu")
			nodeBranchA.SetValueType(ir.ValueType{
				Shape:       shape,
				DType:       dtype.Float64,
				Layout:      ir.LayoutRowMajor,
				MemoryClass: ir.MemoryDevice,
			})
			nodeBranchA.SetAlias(ir.Alias{Kind: ir.AliasInput, InputIndex: 0})
			nodeBranchA.SetInPlace(true)
			nodeBranchA.SetAttribute("gain", ir.IntAttribute(2))
			nodeBranchA.SetMetadata("semantic", "kept")

			optimized, err := optimizer.Optimize(graph)
			So(err, ShouldBeNil)

			var branch *ir.Node
			for _, node := range optimized.Nodes() {
				if node.ID() == "branch_a" {
					branch = node
				}
			}

			So(branch, ShouldNotBeNil)
			So(branch.OperationID(), ShouldEqual, ir.OpID("custom.relu"))
			So(branch.ValueType().Layout, ShouldEqual, ir.LayoutRowMajor)
			So(branch.ValueType().MemoryClass, ShouldEqual, ir.MemoryDevice)
			So(branch.Alias(), ShouldResemble, ir.Alias{Kind: ir.AliasInput, InputIndex: 0})
			So(branch.InPlace(), ShouldBeTrue)
			So(branch.Attribute("gain"), ShouldEqual, ir.IntAttribute(2))
			So(branch.Metadata()["semantic"], ShouldEqual, "kept")
		})
	})

	Convey("Given equivalent commutative expressions with different input order", t, func() {
		optimizer := NewCSEOptimizer()
		graph := ir.NewGraph()

		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		left := ir.NewNode("left", ir.OpInput, shape)
		right := ir.NewNode("right", ir.OpInput, shape)

		first := ir.NewNode("first", ir.OpAdd, shape)
		first.AddInput(left)
		first.AddInput(right)

		second := ir.NewNode("second", ir.OpAdd, shape)
		second.AddInput(right)
		second.AddInput(left)

		sink := ir.NewNode("sink", ir.OpMul, shape)
		sink.AddInput(first)
		sink.AddInput(second)

		graph.AddNode(left)
		graph.AddNode(right)
		graph.AddNode(first)
		graph.AddNode(second)
		graph.AddNode(sink)

		Convey("It should fold them through replacement IDs instead of original input order", func() {
			optimized, err := optimizer.Optimize(graph)
			So(err, ShouldBeNil)

			nodes := optimized.Nodes()
			So(len(nodes), ShouldEqual, 4)

			var optimizedSink *ir.Node
			for _, node := range nodes {
				if node.ID() == "sink" {
					optimizedSink = node
				}
			}

			So(optimizedSink, ShouldNotBeNil)
			So(len(optimizedSink.Inputs()), ShouldEqual, 2)
			So(optimizedSink.Inputs()[0].ID(), ShouldEqual, optimizedSink.Inputs()[1].ID())
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
	for b.Loop() {
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
