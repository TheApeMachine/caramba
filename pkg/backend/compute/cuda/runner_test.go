package cuda

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestRunner(t *testing.T) {
	Convey("Given a CUDA Runner", t, func() {
		ctx := context.Background()
		runner := NewRunner()

		Convey("Location should be CUDA", func() {
			So(runner.Location(), ShouldEqual, tensor.CUDA)
		})

		Convey("Execute should fail if no targets provided", func() {
			graph := ir.NewGraph()
			_, err := runner.Execute(ctx, graph, nil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no execution targets provided")
		})

		Convey("Execute should succeed with valid targets", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{2, 2})
			So(err, ShouldBeNil)

			node := ir.NewNode("in", ir.OpInput, shape)
			node.SetMetadata("values", []float64{1, 2, 3, 4})
			graph.AddNode(node)

			results, err := runner.Execute(ctx, graph, []*ir.Node{node})

			if err != nil {
				So(err.Error(), ShouldContainSubstring, cudaTensorUnavailableMsg)

				return
			}

			values, err := tensorFloat64Values(results["in"])
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3, 4})
		})
	})
}

func tensorFloat64Values(input tensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := input.RawBytes()
	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
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
			_, err := runner.Execute(ctx, graph, targets)
			if err != nil {
				b.Fatalf("Execute failed: %v", err)
			}
		}
	})
}
