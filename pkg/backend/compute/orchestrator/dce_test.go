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

		Convey("It should stop when context is cancelled", func() {
			cancelledContext, cancel := context.WithCancel(context.Background())
			cancel()

			optimized, err := optimizer.Optimize(cancelledContext, graph, []*ir.Node{nodeSink})

			So(err, ShouldEqual, context.Canceled)
			So(optimized, ShouldBeNil)
		})

		Convey("It should preserve dependencies for side-effecting nodes", func() {
			nodeSideEffectInput := ir.NewNode("side_effect_input", ir.OpInput, shape)
			nodeSideEffect := ir.NewNode("side_effect", ir.OpAdd, shape)
			nodeSideEffect.SetOperationID("custom.state.write")
			nodeSideEffect.SetValueType(ir.ValueType{
				Shape:       shape,
				DType:       tensor.Float64,
				Layout:      ir.LayoutColumnMajor,
				MemoryClass: ir.MemoryUnified,
			})
			nodeSideEffect.SetEffect(ir.EffectStateWrite)
			nodeSideEffect.SetAlias(ir.Alias{Kind: ir.AliasInput, InPlace: true, InputIndex: 0})
			nodeSideEffect.SetInPlace(true)
			nodeSideEffect.SetAttribute("checkpoint", ir.BoolAttribute(true))
			nodeSideEffect.SetMetadata("semantic", "kept")
			nodeSideEffect.AddInput(nodeSideEffectInput)

			graph.AddNode(nodeSideEffectInput)
			graph.AddNode(nodeSideEffect)

			optimized, err := optimizer.Optimize(ctx, graph, []*ir.Node{nodeSink})
			So(err, ShouldBeNil)

			var ids []string
			var sideEffect *ir.Node

			for _, node := range optimized.Nodes() {
				ids = append(ids, node.ID())

				if node.ID() == "side_effect" {
					sideEffect = node
				}
			}

			So(ids, ShouldContain, "side_effect_input")
			So(ids, ShouldContain, "side_effect")
			So(sideEffect, ShouldNotBeNil)
			So(sideEffect.Inputs()[0].ID(), ShouldEqual, "side_effect_input")
			So(sideEffect.OperationID(), ShouldEqual, ir.OpID("custom.state.write"))
			So(sideEffect.ValueType().Layout, ShouldEqual, ir.LayoutColumnMajor)
			So(sideEffect.ValueType().MemoryClass, ShouldEqual, ir.MemoryUnified)
			So(sideEffect.Effect(), ShouldEqual, ir.EffectStateWrite)
			So(sideEffect.Alias(), ShouldResemble, ir.Alias{Kind: ir.AliasInput, InPlace: true, InputIndex: 0})
			So(sideEffect.InPlace(), ShouldBeTrue)
			So(sideEffect.Attribute("checkpoint"), ShouldEqual, ir.BoolAttribute(true))
			So(sideEffect.Metadata()["semantic"], ShouldEqual, "kept")
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
	for b.Loop() {
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
