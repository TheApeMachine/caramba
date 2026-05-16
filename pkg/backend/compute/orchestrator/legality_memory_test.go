package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestLegalityAwareFusion(t *testing.T) {
	Convey("Given fusion legality from backend capabilities", t, func() {
		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		left := ir.NewNode("left", ir.OpInput, shape)
		right := ir.NewNode("right", ir.OpInput, shape)
		matmul := ir.NewNode("matmul", ir.OpMatmul, shape)
		matmul.AddInput(left)
		matmul.AddInput(right)
		gelu := ir.NewNode("gelu", ir.OpGELU, shape)
		gelu.AddInput(matmul)
		graph.AddNode(left)
		graph.AddNode(right)
		graph.AddNode(matmul)
		graph.AddNode(gelu)

		Convey("It should not fuse when the selected backend rejects the fused pattern", func() {
			capabilities := NewStaticCapabilities(tensor.Host)
			capabilities.Register(ir.OpInput)
			capabilities.Register(ir.OpMatmul)
			capabilities.Register(ir.OpGELU)
			optimizer := NewFusionOptimizerWithCapabilities(capabilities)

			optimized, err := optimizer.Optimize(graph)

			So(err, ShouldBeNil)
			So(optimized.Nodes(), ShouldHaveLength, 4)
		})

		Convey("It should fuse when the selected backend accepts the fused pattern", func() {
			capabilities := NewStaticCapabilities(tensor.Host)
			capabilities.Register(ir.OpInput)
			capabilities.Register(ir.OpMatmul)
			capabilities.Register(ir.OpGELU)
			capabilities.RegisterFusion("matmul.activation", ir.OpFused)
			optimizer := NewFusionOptimizerWithCapabilities(capabilities)

			optimized, err := optimizer.Optimize(graph)

			So(err, ShouldBeNil)
			So(optimized.Nodes(), ShouldHaveLength, 3)
		})
	})
}

func TestBackendCapabilities(t *testing.T) {
	Convey("Given backend capability declarations", t, func() {
		Convey("It should expose resident backend support without wildcard fallbacks", func() {
			So(CapabilitiesForLocation(tensor.Host).Supports("attention.sdpa"), ShouldBeTrue)
			So(CapabilitiesForLocation(tensor.Host).Supports("*"), ShouldBeFalse)
			So(CapabilitiesForLocation(tensor.CUDA).Supports("attention.sdpa"), ShouldBeFalse)
			So(CapabilitiesForLocation(tensor.CUDA).Supports(ir.OpMatmul), ShouldBeTrue)
		})

		Convey("It should expose Metal native operation families explicitly", func() {
			metal := CapabilitiesForLocation(tensor.Metal)

			So(metal.Supports("attention.sdpa"), ShouldBeTrue)
			So(metal.Supports("projection.linear"), ShouldBeTrue)
			So(metal.Supports("vsa.bind"), ShouldBeFalse)
			So(metal.Supports("train.optimizer.adam"), ShouldBeFalse)
			So(metal.Precision("attention.sdpa"), ShouldEqual, tensor.Float32)
		})
	})
}

func TestLoweringPass_Precision(t *testing.T) {
	Convey("Given Metal f32 capabilities", t, func() {
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		node := ir.NewNode("relu", ir.OpReLU, shape)
		graph.AddNode(node)

		input := PassInput{
			Graph:       graph,
			Targets:     []*ir.Node{node},
			TargetMap:   targetMap([]*ir.Node{node}),
			Diagnostics: &Diagnostics{},
		}

		Convey("It should reject default f64 precision requirements", func() {
			_, err := NewLoweringPass(CapabilitiesForLocation(tensor.Metal)).Run(context.Background(), input)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "requires float64 precision")
		})

		Convey("It should allow explicit f32 precision opt-in", func() {
			node.SetValueType(ir.ValueType{Shape: shape, DType: tensor.Float64, Precision: tensor.Float32})

			_, err := NewLoweringPass(CapabilitiesForLocation(tensor.Metal)).Run(context.Background(), input)

			So(err, ShouldBeNil)
		})
	})
}

func TestEffectAwareDCE(t *testing.T) {
	Convey("Given side-effect-aware DCE", t, func() {
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		input := ir.NewNode("input", ir.OpInput, shape)
		target := ir.NewNode("target", ir.OpReLU, shape)
		target.AddInput(input)
		checkpoint := ir.NewNode("checkpoint", "train.checkpoint.save", shape)
		checkpoint.SetEffect(ir.EffectStateWrite)
		checkpoint.AddInput(input)
		deadPure := ir.NewNode("dead", ir.OpGELU, shape)
		deadPure.AddInput(input)
		graph.AddNode(input)
		graph.AddNode(target)
		graph.AddNode(checkpoint)
		graph.AddNode(deadPure)

		optimized, err := NewDCEOptimizer().Optimize(context.Background(), graph, []*ir.Node{target})

		So(err, ShouldBeNil)
		So(nodeIDs(optimized), ShouldContain, "checkpoint")
		So(nodeIDs(optimized), ShouldNotContain, "dead")
	})
}

func TestMemoryPlanner(t *testing.T) {
	Convey("Given a graph memory planner", t, func() {
		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		input := ir.NewNode("input", ir.OpInput, shape)
		left := ir.NewNode("left", ir.OpReLU, shape)
		left.AddInput(input)
		right := ir.NewNode("right", ir.OpGELU, shape)
		right.AddInput(input)
		output := ir.NewNode("output", ir.OpAdd, shape)
		output.AddInput(left)
		output.AddInput(right)
		graph.AddNode(input)
		graph.AddNode(left)
		graph.AddNode(right)
		graph.AddNode(output)

		plan, err := NewMemoryPlanner().Plan(graph, []*ir.Node{output})

		So(err, ShouldBeNil)
		So(plan.Lifetime("input").FirstUse, ShouldEqual, 0)
		So(plan.Lifetime("input").LastUse, ShouldBeGreaterThan, plan.Lifetime("input").FirstUse)
		So(plan.Buffer("left"), ShouldNotEqual, "")
		So(plan.Buffer("right"), ShouldNotEqual, "")
	})
}

func TestSchedulePass(t *testing.T) {
	Convey("Given a cost-aware schedule pass", t, func() {
		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		left := ir.NewNode("left", ir.OpInput, shape)
		right := ir.NewNode("right", ir.OpInput, shape)
		output := ir.NewNode("output", ir.OpMatmul, shape)
		output.AddInput(left)
		output.AddInput(right)
		graph.AddNode(left)
		graph.AddNode(right)
		graph.AddNode(output)

		result, err := NewPipeline(NewSchedulePass(CapabilitiesForLocation(tensor.Host))).Run(
			context.Background(),
			graph,
			[]*ir.Node{output},
		)

		So(err, ShouldBeNil)
		So(result.Graph.Nodes()[2].Metadata()["schedule_layer"], ShouldEqual, 1)
		So(result.Graph.Nodes()[2].Metadata()["estimated_flops"], ShouldBeGreaterThan, uint64(0))
	})
}

func nodeIDs(graph *ir.Graph) []string {
	ids := make([]string, 0, len(graph.Nodes()))

	for _, node := range graph.Nodes() {
		ids = append(ids, node.ID())
	}

	return ids
}
