package cpu

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestRunner(t *testing.T) {
	Convey("Given a CPU Runner", t, func() {
		ctx := context.Background()
		runner := NewRunner()

		Convey("Location should be Host", func() {
			So(runner.Location(), ShouldEqual, tensor.Host)
		})

		Convey("Execute should fail if no targets provided", func() {
			graph := ir.NewGraph()
			_, err := runner.Execute(ctx, graph, nil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no execution targets provided")
		})

		Convey("Execute should fail if graph is nil", func() {
			shape, err := tensor.NewShape([]int{1})
			So(err, ShouldBeNil)

			target := ir.NewNode("target", ir.OpInput, shape)
			_, err = runner.Execute(ctx, nil, []*ir.Node{target})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "graph is required")
		})

		Convey("Execute should fail if a target is not registered", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{1})
			So(err, ShouldBeNil)

			target := ir.NewNode("target", ir.OpInput, shape)
			_, err = runner.Execute(ctx, graph, []*ir.Node{target})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "target node \"target\" is not registered")
		})

		Convey("Execute should succeed with valid targets", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{2, 2})
			So(err, ShouldBeNil)

			node := ir.NewNode("in", ir.OpInput, shape)
			node.SetMetadata("values", []float64{1, 2, 3, 4})
			graph.AddNode(node)

			results, err := runner.Execute(ctx, graph, []*ir.Node{node})
			So(err, ShouldBeNil)
			So(results, ShouldNotBeNil)
			values, err := tensorFloat64Values(results["in"])
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3, 4})
		})

		Convey("Execute should dispatch CPU kernels through the executor", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{2, 2})
			So(err, ShouldBeNil)

			left := ir.NewNode("left", ir.OpInput, shape)
			left.SetMetadata("values", []float64{1, 2, 3, 4})
			right := ir.NewNode("right", ir.OpInput, shape)
			right.SetMetadata("values", []float64{5, 6, 7, 8})
			mul := ir.NewNode("mul", ir.OpMul, shape)
			mul.AddInput(left)
			mul.AddInput(right)
			relu := ir.NewNode("relu", ir.OpReLU, shape)
			relu.AddInput(mul)

			graph.AddNode(left)
			graph.AddNode(right)
			graph.AddNode(mul)
			graph.AddNode(relu)

			results, err := runner.Execute(ctx, graph, []*ir.Node{relu})
			So(err, ShouldBeNil)

			values, err := tensorFloat64Values(results["relu"])
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{5, 12, 21, 32})
		})

		Convey("Execute should not leak intermediates across repeated runs", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{64})
			So(err, ShouldBeNil)

			input := ir.NewNode("input", ir.OpInput, shape)
			input.SetMetadata("values", make([]float64, shape.Len()))
			first := ir.NewNode("first", ir.OpReLU, shape)
			first.AddInput(input)
			second := ir.NewNode("second", ir.OpGELU, shape)
			second.AddInput(first)
			graph.AddNode(input)
			graph.AddNode(first)
			graph.AddNode(second)

			for range 256 {
				results, err := runner.Execute(ctx, graph, []*ir.Node{second})
				So(err, ShouldBeNil)
				So(results["second"].Close(), ShouldBeNil)
			}
		})

		Convey("Execute should fail after Close", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{1})
			So(err, ShouldBeNil)

			node := ir.NewNode("in", ir.OpInput, shape)
			node.SetMetadata("values", []float64{1})
			graph.AddNode(node)

			So(runner.Close(), ShouldBeNil)
			results, err := runner.Execute(ctx, graph, []*ir.Node{node})

			So(err, ShouldNotBeNil)
			So(results, ShouldBeNil)
		})
	})
}

func BenchmarkRunner(b *testing.B) {
	ctx := context.Background()
	runner := NewRunner()
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	b.Run("SimpleGraph", func(b *testing.B) {
		graph := ir.NewGraph()
		node := ir.NewNode("in", ir.OpInput, shape)
		node.SetMetadata("values", []float64{1, 2, 3, 4})
		graph.AddNode(node)
		targets := []*ir.Node{node}

		b.ResetTimer()
		for b.Loop() {
			outputs, err := runner.Execute(ctx, graph, targets)
			if err != nil {
				b.Fatalf("Execute failed: %v", err)
			}

			for _, output := range outputs {
				if err := output.Close(); err != nil {
					b.Fatalf("Close failed: %v", err)
				}
			}
		}
	})

	b.Run("ComplexGraph", func(b *testing.B) {
		graph := ir.NewGraph()
		nodeIn := ir.NewNode("in", ir.OpInput, shape)
		nodeIn.SetMetadata("values", []float64{-1, 2, -3, 4})
		nodeMath := ir.NewNode("relu", ir.OpReLU, shape)
		nodeMath.AddInput(nodeIn)
		graph.AddNode(nodeIn)
		graph.AddNode(nodeMath)
		targets := []*ir.Node{nodeMath}

		b.ResetTimer()
		for b.Loop() {
			outputs, err := runner.Execute(ctx, graph, targets)
			if err != nil {
				b.Fatalf("Execute failed: %v", err)
			}

			for _, output := range outputs {
				if err := output.Close(); err != nil {
					b.Fatalf("Close failed: %v", err)
				}
			}
		}
	})
}
