package xla

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestRunner(t *testing.T) {
	Convey("Given a XLA Runner", t, func() {
		ctx := context.Background()
		runner := NewRunner()

		Convey("Location should be XLA", func() {
			So(runner.Location(), ShouldEqual, tensor.XLA)
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
			graph.AddNode(node)

			results, err := runner.Execute(ctx, graph, []*ir.Node{node})
			So(err, ShouldBeNil)
			So(results, ShouldNotBeNil)
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
		graph.AddNode(node)
		targets := []*ir.Node{node}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := runner.Execute(ctx, graph, targets)
			if err != nil {
				b.Fatalf("Execute failed: %v", err)
			}
		}
	})
}
