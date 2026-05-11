package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

const tanhApproxTol = 5e-2

var benchSinkTanh []float64

func TestTanh(t *testing.T) {
	Convey("Given a Tanh operation", t, func() {
		op := NewTanh()

		Convey("Forward", func() {
			Convey(
				"When length is not a multiple of SIMD width, scalar tail handling runs after any vector prefix",
				func() {
					inputs := []float64{-1, 0, 1}
					out := op.Forward([]int{len(inputs)}, inputs)

					for index := range inputs {
						So(out[index], ShouldAlmostEqual, math.Tanh(inputs[index]), tanhApproxTol)
					}
				},
			)

			Convey("It should stay close to math.Tanh on ±1 (rational SIMD differs slightly from libm)", func() {
				inputs := []float64{-1, 1}
				out := op.Forward([]int{len(inputs)}, inputs)

				So(out[0], ShouldAlmostEqual, math.Tanh(inputs[0]), tanhApproxTol)
				So(out[1], ShouldAlmostEqual, math.Tanh(inputs[1]), tanhApproxTol)
			})

			Convey("It should return an empty slice for empty input", func() {
				So(op.Forward([]int{0}, []float64{}), ShouldBeEmpty)
			})

			Convey(
				"It should map large magnitude inputs toward ±1 (approximation quality varies by SIMD path)",
				func() {
					inputs := []float64{-10, 10}
					out := op.Forward([]int{len(inputs)}, inputs)

					So(out[0], ShouldBeLessThan, 0)
					So(out[1], ShouldBeGreaterThan, 0)
					So(math.Abs(out[0]), ShouldBeGreaterThan, 0.99)
					So(math.Abs(out[1]), ShouldBeGreaterThan, 0.99)
				},
			)

			Convey("It should map ±Inf to ±1", func() {
				outPos := op.Forward([]int{1}, []float64{math.Inf(1)})
				outNeg := op.Forward([]int{1}, []float64{math.Inf(-1)})

				So(outPos[0], ShouldAlmostEqual, 1, 1e-12)
				So(outNeg[0], ShouldAlmostEqual, -1, 1e-12)
			})

			Convey("It should propagate NaN", func() {
				out := op.Forward([]int{1}, []float64{math.NaN()})

				So(math.IsNaN(out[0]), ShouldBeTrue)
			})
		})
	})
}

func BenchmarkTanh_Forward(benchmark *testing.B) {
	op := NewTanh()
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512)/256 - 1
	}

	shape := []int{len(input)}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		benchSinkTanh = op.Forward(shape, input)
	}

	_ = benchSinkTanh
}
