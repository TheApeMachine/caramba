package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestOptimizerNumericalEquivalence(t *testing.T) {
	Convey("Given an optimizable graph", t, func() {
		ctx := context.Background()
		graph, target := optimizerProofGraph(t)

		Convey("It should produce the same target values before and after optimization", func() {
			directRunner := cpu.NewRunner()
			defer directRunner.Close()
			optimizedRunner := cpu.NewRunner()
			defer optimizedRunner.Close()

			expected, err := directRunner.Execute(ctx, graph, []*ir.Node{target})
			So(err, ShouldBeNil)

			scheduler := NewScheduler()
			scheduler.RegisterRunner(optimizedRunner)
			actual, err := scheduler.Execute(ctx, graph, []*ir.Node{target}, tensor.Host)
			So(err, ShouldBeNil)

			expectedValues, err := expected[target.ID()].CloneFloat64()
			So(err, ShouldBeNil)
			actualValues, err := actual["matmul_fused_activation"].CloneFloat64()
			So(err, ShouldBeNil)
			So(actualValues, ShouldResemble, expectedValues)
		})
	})
}

func BenchmarkOptimizerProofMLP(b *testing.B) {
	ctx := context.Background()
	scheduler := NewScheduler()
	runner := cpu.NewRunner()
	defer runner.Close()
	scheduler.RegisterRunner(runner)

	graph, target := optimizerProofGraph(b)

	for b.Loop() {
		_, err := scheduler.Execute(ctx, graph, []*ir.Node{target}, tensor.Host)
		if err != nil {
			b.Fatalf("scheduler execute failed: %v", err)
		}
	}
}

type testingFatal interface {
	Fatalf(format string, args ...any)
}

func optimizerProofGraph(testingObject testingFatal) (*ir.Graph, *ir.Node) {
	shape, err := tensor.NewShape([]int{2, 2})
	if err != nil {
		testingObject.Fatalf("NewShape failed: %v", err)
	}

	graph := ir.NewGraph()
	left := ir.NewNode("left", ir.OpInput, shape)
	left.SetMetadata("values", []float64{1, 2, 3, 4})
	right := ir.NewNode("right", ir.OpInput, shape)
	right.SetMetadata("values", []float64{1, 0, 0, 1})
	matmul := ir.NewNode("matmul", ir.OpMatmul, shape)
	matmul.AddInput(left)
	matmul.AddInput(right)
	activation := ir.NewNode("activation", ir.OpReLU, shape)
	activation.AddInput(matmul)
	graph.AddNode(left)
	graph.AddNode(right)
	graph.AddNode(matmul)
	graph.AddNode(activation)

	return graph, activation
}
