package cpu

import (
	"context"
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestTensorBackend_Apply(t *testing.T) {
	Convey("Given a CPU tensor backend", t, func() {
		tensorBackend := NewTensorBackend()

		Convey("It should route elementwise operations through state.Dict operations", func() {
			left := uploadTestTensor(tensorBackend, []int{3}, []float64{1, 2, 3})
			right := uploadTestTensor(tensorBackend, []int{3}, []float64{4, 5, 6})
			defer func() { So(left.Close(), ShouldBeNil) }()
			defer func() { So(right.Close(), ShouldBeNil) }()

			output, err := tensorBackend.Apply(context.Background(), executor.NodeSpec{
				ID:    "add",
				Op:    ir.OpAdd,
				Shape: []int{3},
			}, []tensor.Float64Tensor{left, right})

			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{5, 7, 9})
		})

		Convey("It should route Swish by normalized operation id", func() {
			input := uploadTestTensor(tensorBackend, []int{3}, []float64{-1, 0, 2})
			defer func() { So(input.Close(), ShouldBeNil) }()

			output, err := tensorBackend.Apply(context.Background(), executor.NodeSpec{
				ID:    "swish",
				Op:    ir.OpSwish,
				Shape: []int{3},
			}, []tensor.Float64Tensor{input})

			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)

			So(values[0], ShouldAlmostEqual, swishApplyReference(-1), 1e-9)
			So(values[1], ShouldAlmostEqual, swishApplyReference(0), 1e-9)
			So(values[2], ShouldAlmostEqual, swishApplyReference(2), 1e-9)
		})

		Convey("It should route SELU by normalized operation id", func() {
			input := uploadTestTensor(tensorBackend, []int{3}, []float64{-1, 0, 2})
			defer func() { So(input.Close(), ShouldBeNil) }()

			output, err := tensorBackend.Apply(context.Background(), executor.NodeSpec{
				ID:    "selu",
				Op:    ir.OpSELU,
				Shape: []int{3},
			}, []tensor.Float64Tensor{input})

			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)

			So(values[0], ShouldAlmostEqual, seluApplyReference(-1), 1e-9)
			So(values[1], ShouldAlmostEqual, seluApplyReference(0), 1e-9)
			So(values[2], ShouldAlmostEqual, seluApplyReference(2), 1e-9)
		})

		Convey("It should derive matmul operation shape from resident inputs", func() {
			left := uploadTestTensor(tensorBackend, []int{2, 3}, []float64{
				1, 2, 3,
				4, 5, 6,
			})
			right := uploadTestTensor(tensorBackend, []int{3, 2}, []float64{
				7, 8,
				9, 10,
				11, 12,
			})
			defer func() { So(left.Close(), ShouldBeNil) }()
			defer func() { So(right.Close(), ShouldBeNil) }()

			output, err := tensorBackend.Apply(context.Background(), executor.NodeSpec{
				ID:    "matmul",
				Op:    ir.OpMatmul,
				Shape: []int{2, 2},
			}, []tensor.Float64Tensor{left, right})

			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()
			So(output.Shape().Dims(), ShouldResemble, []int{2, 2})

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{58, 64, 139, 154})
		})

		Convey("It should execute fused matmul, bias, and activation through operations", func() {
			left := uploadTestTensor(tensorBackend, []int{1, 2}, []float64{1, -1})
			right := uploadTestTensor(tensorBackend, []int{2, 2}, []float64{
				1, 2,
				3, 4,
			})
			bias := uploadTestTensor(tensorBackend, []int{2}, []float64{0, 1})
			defer func() { So(left.Close(), ShouldBeNil) }()
			defer func() { So(right.Close(), ShouldBeNil) }()
			defer func() { So(bias.Close(), ShouldBeNil) }()

			output, err := tensorBackend.Apply(context.Background(), executor.NodeSpec{
				ID:    "fused",
				Op:    ir.OpFused,
				Shape: []int{1, 2},
				Metadata: map[string]any{
					"activation": string(ir.OpGELU),
				},
			}, []tensor.Float64Tensor{left, right, bias})

			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values[0], ShouldAlmostEqual, geluApplyReference(-2), 1e-9)
			So(values[1], ShouldAlmostEqual, geluApplyReference(-1), 1e-9)
		})
	})
}

func BenchmarkTensorBackend_ApplyMatmul(benchmark *testing.B) {
	tensorBackend := NewTensorBackend()
	leftValues := make([]float64, 64*64)
	rightValues := make([]float64, 64*64)

	for index := range leftValues {
		leftValues[index] = float64(index%17) * 0.125
		rightValues[index] = float64(index%19) * 0.0625
	}

	left := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64, 64}, leftValues)
	right := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64, 64}, rightValues)
	defer func() { _ = left.Close() }()
	defer func() { _ = right.Close() }()

	node := executor.NodeSpec{
		ID:    "matmul",
		Op:    ir.OpMatmul,
		Shape: []int{64, 64},
	}

	for benchmark.Loop() {
		output, err := tensorBackend.Apply(context.Background(), node, []tensor.Float64Tensor{left, right})

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func geluApplyReference(value float64) float64 {
	cube := value * value * value
	inner := 0.7978845608028654 * (value + 0.044715*cube)

	return 0.5 * value * (1 + stdmath.Tanh(inner))
}

func swishApplyReference(value float64) float64 {
	return value / (1 + stdmath.Exp(-value))
}

func seluApplyReference(value float64) float64 {
	if value > 0 {
		return 1.0507009873554805 * value
	}

	return 1.7580993408473766 * (stdmath.Exp(value) - 1)
}
