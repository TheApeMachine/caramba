package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

// MockRunner is a test stub for runner.Runner
type MockRunner struct {
	executed bool
}

func (m *MockRunner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	m.executed = true
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
		scheduler := NewScheduler(ctx)

		mockRunner := &MockRunner{}
		scheduler.RegisterRunner(mockRunner)

		Convey("It should execute on the registered runner", func() {
			graph := ir.NewGraph(ctx)
			shape, _ := tensor.NewShape([]int{1})
			graph.AddNode(ir.NewNode(ctx, "a", ir.OpInput, shape))

			_, err := scheduler.Execute(graph, nil, tensor.Host)

			So(err, ShouldBeNil)
			So(mockRunner.executed, ShouldBeTrue)
		})

		Convey("It should fail when location is missing", func() {
			graph := ir.NewGraph(ctx)
			_, err := scheduler.Execute(graph, nil, tensor.CUDA)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no runner registered")
		})
	})
}

func BenchmarkScheduler(b *testing.B) {
	ctx := context.Background()
	scheduler := NewScheduler(ctx)

	mockRunner := &MockRunner{}
	scheduler.RegisterRunner(mockRunner)

	graph := ir.NewGraph(ctx)
	shape, _ := tensor.NewShape([]int{1})
	graph.AddNode(ir.NewNode(ctx, "a", ir.OpInput, shape))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scheduler.Execute(graph, nil, tensor.Host)
	}
}
