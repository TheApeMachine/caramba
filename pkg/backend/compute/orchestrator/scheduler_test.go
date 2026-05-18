package orchestrator

import (
	"context"
	"sync/atomic"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type MockRunner struct {
	executed atomic.Bool
	targets  atomic.Pointer[[]*ir.Node]
}

func (m *MockRunner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Tensor, error) {
	m.executed.Store(true)
	m.targets.Store(&targets)

	return make(map[string]tensor.Tensor), nil
}

func (m *MockRunner) Location() tensor.Location {
	return tensor.Host
}

func (m *MockRunner) Close() error {
	return nil
}

func TestScheduler(t *testing.T) {
	Convey("Given a Scheduler", t, func() {
		ctx := context.Background()
		scheduler := NewScheduler()

		mockRunner := &MockRunner{}
		scheduler.RegisterRunner(mockRunner)

		Convey("It should execute on the registered runner", func() {
			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{1})
			So(err, ShouldBeNil)
			graph.AddNode(ir.NewNode("a", ir.OpInput, shape))

			results, err := scheduler.Execute(ctx, graph, nil, tensor.Host)

			So(err, ShouldBeNil)
			So(results, ShouldNotBeNil)
			So(mockRunner.executed.Load(), ShouldBeTrue)
		})

		Convey("It should fail when location is missing", func() {
			graph := ir.NewGraph()
			_, err := scheduler.Execute(ctx, graph, nil, tensor.CUDA)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no runner registered")
		})

		Convey("It should stop before pipeline work when context is cancelled", func() {
			cancelledContext, cancel := context.WithCancel(context.Background())
			cancel()

			graph := ir.NewGraph()
			shape, err := tensor.NewShape([]int{1})
			So(err, ShouldBeNil)
			graph.AddNode(ir.NewNode("a", ir.OpInput, shape))

			results, err := scheduler.Execute(cancelledContext, graph, nil, tensor.Host)

			So(err, ShouldEqual, context.Canceled)
			So(results, ShouldBeNil)
			So(mockRunner.executed.Load(), ShouldBeFalse)
		})

		Convey("It should remap explicit targets after fusion", func() {
			shape, err := tensor.NewShape([]int{1, 1})
			So(err, ShouldBeNil)

			graph := ir.NewGraph()
			left := ir.NewNode("left", ir.OpInput, shape)
			right := ir.NewNode("right", ir.OpInput, shape)
			matmul := ir.NewNode("matmul", ir.OpMatmul, shape)
			matmul.AddInput(left)
			matmul.AddInput(right)
			relu := ir.NewNode("relu", ir.OpReLU, shape)
			relu.AddInput(matmul)
			dead := ir.NewNode("dead", ir.OpGELU, shape)
			dead.AddInput(left)
			graph.AddNode(left)
			graph.AddNode(right)
			graph.AddNode(matmul)
			graph.AddNode(relu)
			graph.AddNode(dead)

			_, err = scheduler.Execute(ctx, graph, []*ir.Node{relu}, tensor.Host)

			So(err, ShouldBeNil)
			executedTargets := mockRunner.targets.Load()
			So(executedTargets, ShouldNotBeNil)
			So(*executedTargets, ShouldHaveLength, 1)
			So((*executedTargets)[0].OpType(), ShouldEqual, ir.OpFused)
		})
	})
}

func BenchmarkScheduler(b *testing.B) {
	ctx := context.Background()
	scheduler := NewScheduler()

	mockRunner := &MockRunner{}
	scheduler.RegisterRunner(mockRunner)

	shape, err := tensor.NewShape([]int{1})
	if err != nil {
		b.Fatalf("NewShape failed: %v", err)
	}

	b.Run("SimpleGraph", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			b.StopTimer()
			graph := ir.NewGraph()
			graph.AddNode(ir.NewNode("a", ir.OpInput, shape))
			b.StartTimer()

			_, err := scheduler.Execute(ctx, graph, nil, tensor.Host)
			if err != nil {
				b.Fatalf("Execute failed: %v", err)
			}
		}
	})
}
