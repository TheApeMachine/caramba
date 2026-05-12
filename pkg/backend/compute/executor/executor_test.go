package executor_test

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestExecutor(t *testing.T) {
	Convey("Given a shared graph executor", t, func() {
		backend := computecpu.NewTensorBackend()
		graphExecutor := executor.New(backend)
		shape, err := tensor.NewShape([]int{2})
		So(err, ShouldBeNil)
		data, err := executor.EncodeFloat64([]float64{-1, 2})
		So(err, ShouldBeNil)

		nodes := []executor.NodeSpec{
			{
				ID:     "input",
				Op:     ir.OpInput,
				Shape:  shape.Dims(),
				Target: false,
			},
			{
				ID:     "relu",
				Op:     ir.OpReLU,
				Inputs: []string{"input"},
				Shape:  shape.Dims(),
				Target: true,
			},
		}

		tensors := []executor.TensorSpec{
			{
				ID:    "input",
				Shape: []int64{2},
				Data:  data,
				DType: tensor.Float64,
			},
		}

		Convey("When executing a supported graph", func() {
			outputs, err := graphExecutor.Execute(context.Background(), nodes, tensors)

			Convey("It should execute through the backend kernels", func() {
				So(err, ShouldBeNil)
				So(outputs, ShouldHaveLength, 1)
				So(outputs[0].ID, ShouldEqual, "relu")

				values, err := executor.DecodeFloat64(outputs[0].Data)
				So(err, ShouldBeNil)
				So(values, ShouldResemble, []float64{0, 2})
			})
		})
	})
}

func BenchmarkExecutor_Execute(benchmark *testing.B) {
	backend := computecpu.NewTensorBackend()
	data, err := executor.EncodeFloat64([]float64{1, 2})
	if err != nil {
		benchmark.Fatalf("EncodeFloat64 failed: %v", err)
	}

	nodes := []executor.NodeSpec{
		{ID: "input", Op: ir.OpInput},
		{ID: "relu", Op: ir.OpReLU, Inputs: []string{"input"}, Target: true},
	}

	tensors := []executor.TensorSpec{
		{ID: "input", Shape: []int64{2}, Data: data, DType: tensor.Float64},
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		graphExecutor := executor.New(backend)
		_, err := graphExecutor.Execute(context.Background(), nodes, tensors)
		if err != nil {
			benchmark.Fatalf("Execute failed: %v", err)
		}
	}
}
