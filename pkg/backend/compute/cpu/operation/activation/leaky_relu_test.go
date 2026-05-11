package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestLeakyReLU(t *testing.T) {
	Convey("Given a LeakyReLU operation", t, func() {
		op := NewLeakyReLU(0.25)

		Convey("Forward", func() {
			Convey("It should pass positive values through unchanged", func() {
				out := op.Forward([]int{3}, []float64{1, 2, 3})

				So(out, ShouldResemble, []float64{1, 2, 3})
			})

			Convey("It should scale negative values by alpha", func() {
				out := op.Forward([]int{3}, []float64{-1, -2, -3})

				So(out, ShouldResemble, []float64{-0.25, -0.5, -0.75})
			})

			Convey("It should map zeros to zeros", func() {
				out := op.Forward([]int{3}, []float64{0, 0, 0})

				So(out, ShouldResemble, []float64{0, 0, 0})
			})

			Convey("It should not produce NaN for large-magnitude inputs", func() {
				out := op.Forward(
					[]int{4}, []float64{1e100, -1e100, 1e-100, -1e-100},
				)

				So(len(out), ShouldEqual, 4)

				for index := range out {
					So(math.IsNaN(out[index]), ShouldBeFalse)
				}
			})

			Convey("It should handle even-length SIMD-aligned prefixes", func() {
				So(op.Forward([]int{4}, []float64{1, -2, 3, -4}), ShouldResemble,
					[]float64{1, -0.5, 3, -1})
			})

			Convey("It should handle a larger contiguous activation map", func() {
				input := make([]float64, 128)

				for index := range input {
					input[index] = float64(index%16 - 8)
				}

				expected := make([]float64, len(input))

				for index := range input {
					value := input[index]

					if value < 0 {
						expected[index] = 0.25 * value
						continue
					}

					expected[index] = value
				}

				out := op.Forward([]int{len(input)}, input)

				So(out, ShouldResemble, expected)
			})

			Convey("It should panic when no input tensors are provided", func() {
				So(func() { op.Forward([]int{}) }, ShouldPanic)
			})

			Convey(
				"It ignores declared shape element count and uses the raw slice length",
				func() {
					out := op.Forward([]int{2}, []float64{1, 2, 3})

					So(out, ShouldResemble, []float64{1, 2, 3})
				},
			)

			Convey(
				"When length is not a multiple of SIMD width it exercises scalar tail handling (amd64: SSE2/AVX2 process two/four lanes; leftover elements run scalarLeakyReLU)",
				func() {
					out := op.Forward([]int{3}, []float64{-2, 0, 4})

					So(out, ShouldResemble, []float64{-0.5, 0, 4})
				},
			)
		})
	})
}

func BenchmarkLeakyReLU_Forward(benchmark *testing.B) {
	// Alpha distinct from unit tests (0.25): benchmarks commonly use 0.01 as in typical defaults.
	op := NewLeakyReLU(0.01)

	// 4097 elements forces a non-vector-aligned remainder vs SSE2 (×2) and AVX2 (×4) widths,
	// so each iteration stresses SIMD prefixes plus scalar tail.
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512) - 256
	}

	shape := []int{len(input)}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		op.Forward(shape, input)
	}
}
