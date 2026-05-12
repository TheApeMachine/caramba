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
		runner := NewRunner(ctx)

		Convey("Location should be Host", func() {
			So(runner.Location(), ShouldEqual, tensor.Host)
		})

		Convey("Execute should fail if no targets provided", func() {
			graph := ir.NewGraph(ctx)
			_, err := runner.Execute(ctx, graph, nil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no execution targets provided")
		})
	})
}

func BenchmarkRunner(b *testing.B) {
	ctx := context.Background()
	runner := NewRunner(ctx)
	graph := ir.NewGraph(ctx)
	shape, _ := tensor.NewShape([]int{2, 2})
	node := ir.NewNode(ctx, "in", ir.OpInput, shape)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner.Execute(ctx, graph, []*ir.Node{node})
	}
}
