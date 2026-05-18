package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
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
			So(metal.Supports("vsa.bind"), ShouldBeTrue)
			So(metal.Supports("vsa.bundle"), ShouldBeTrue)
			So(metal.Supports("vsa.similarity"), ShouldBeTrue)
			So(metal.Supports("vsa.permute"), ShouldBeTrue)
			So(metal.Supports("vsa.inverse_permute"), ShouldBeTrue)
			So(metal.Supports("markov_blanket.partition"), ShouldBeTrue)
			So(metal.Supports("markov_blanket.flow_internal"), ShouldBeTrue)
			So(metal.Supports("markov_blanket.flow_active"), ShouldBeTrue)
			So(metal.Supports("markov_blanket.mutual_information"), ShouldBeTrue)
			So(metal.Supports("causal.counterfactual"), ShouldBeTrue)
			So(metal.Supports("causal.frontdoor_adjustment"), ShouldBeTrue)
			So(metal.Supports("causal.backdoor_adjustment"), ShouldBeTrue)
			So(metal.Supports("causal.cate"), ShouldBeTrue)
			So(metal.Supports("causal.iv_estimate"), ShouldBeTrue)
			So(metal.Supports("causal.dag_markov_factorization"), ShouldBeTrue)
			So(metal.Supports("causal.do_calculus"), ShouldBeTrue)
			So(metal.Supports("train.loss.mse"), ShouldBeTrue)
			So(metal.Supports("train.loss.cross_entropy"), ShouldBeTrue)
			So(metal.Supports("train.loss.mse_grad"), ShouldBeTrue)
			So(metal.Supports("train.loss.cross_entropy_grad"), ShouldBeTrue)
			So(metal.Supports("bench.accuracy"), ShouldBeTrue)
			So(metal.Supports("bench.metric.accuracy"), ShouldBeTrue)
			So(metal.Supports("bench.perplexity"), ShouldBeTrue)
			So(metal.Supports("bench.metric.perplexity"), ShouldBeTrue)
			So(metal.Supports("bench.f1"), ShouldBeTrue)
			So(metal.Supports("bench.metric.f1"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.adam"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.adamw"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.adamax"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.sgd"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.lion"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.rmsprop"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.hebbian"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.lars"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.lamb"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.adagrad"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.adadelta"), ShouldBeTrue)
			So(metal.Supports("train.optimizer.lbfgs"), ShouldBeTrue)
			So(metal.Precision("attention.sdpa"), ShouldEqual, dtype.Float32)
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
			So(err.Error(), ShouldContainSubstring, "requires F64 precision")
		})

		Convey("It should allow explicit f32 precision opt-in", func() {
			node.SetValueType(ir.ValueType{Shape: shape, DType: dtype.Float64, Precision: dtype.Float32})

			_, err := NewLoweringPass(CapabilitiesForLocation(tensor.Metal)).Run(context.Background(), input)

			So(err, ShouldBeNil)
		})

		Convey("It should reject Metal shapes outside the resident kernel contract", func() {
			leftShape, err := tensor.NewShape([]int{1, 2, 3})
			So(err, ShouldBeNil)
			rightShape, err := tensor.NewShape([]int{3, 2})
			So(err, ShouldBeNil)
			outputShape, err := tensor.NewShape([]int{1, 2})
			So(err, ShouldBeNil)

			valueType := ir.ValueType{
				DType:     dtype.Float64,
				Precision: dtype.Float32,
			}

			graph := ir.NewGraph()
			left := ir.NewNode("left", ir.OpInput, leftShape)
			left.SetValueType(ir.ValueType{Shape: leftShape, DType: dtype.Float64, Precision: dtype.Float32})
			right := ir.NewNode("right", ir.OpInput, rightShape)
			right.SetValueType(ir.ValueType{Shape: rightShape, DType: dtype.Float64, Precision: dtype.Float32})
			output := ir.NewNode("output", "math.matmul", outputShape)
			valueType.Shape = outputShape
			output.SetValueType(valueType)
			output.AddInput(left)
			output.AddInput(right)
			graph.AddNode(left)
			graph.AddNode(right)
			graph.AddNode(output)

			shapeInput := PassInput{
				Graph:       graph,
				Targets:     []*ir.Node{output},
				TargetMap:   targetMap([]*ir.Node{output}),
				Diagnostics: &Diagnostics{},
			}

			_, err = NewLoweringPass(CapabilitiesForLocation(tensor.Metal)).Run(context.Background(), shapeInput)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "requires rank-2 matmul shapes")
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
