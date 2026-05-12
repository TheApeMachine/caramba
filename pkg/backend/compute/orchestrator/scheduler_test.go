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
}

func (m *MockRunner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	m.executed.Store(true)
	return make(map[string]tensor.Float64Tensor), nil
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
		for i := 0; i < b.N; i++ {
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
