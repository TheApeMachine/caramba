package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSigmoid(t *testing.T) {
	Convey("Given a Sigmoid operation", t, func() {
		op := NewSigmoid()

		Convey("Forward", func() {
			Convey(
				"When length is not a multiple of SIMD width (amd64: SSE2×2 / AVX2×4; arm64 NEON×2) scalar tail handling runs after any vector prefix",
				func() {
					out := op.Forward([]int{3}, []float64{-1, 0, 1})

					So(out[0], ShouldBeLessThan, out[1])
					So(out[1], ShouldAlmostEqual, 0.5, 1e-9)
					So(out[2], ShouldBeGreaterThan, out[1])
				},
			)

			Convey("It should treat even-length vectors consistently", func() {
				out := op.Forward([]int{4}, []float64{-2, -1, 1, 2})

				So(out[0], ShouldBeLessThan, out[1])
				So(out[1], ShouldBeLessThan, 0.5)
				So(out[2], ShouldBeGreaterThan, 0.5)
				So(out[3], ShouldBeGreaterThan, out[2])
			})

			Convey("It should return an empty slice for empty input", func() {
				So(op.Forward([]int{0}), ShouldBeEmpty)
			})

			Convey("It should handle a single element", func() {
				So(op.Forward([]int{1}, []float64{0})[0], ShouldAlmostEqual, 0.5, 1e-9)
			})

			Convey("It should approach 0 and 1 for large magnitude inputs", func() {
				pos := op.Forward([]int{1}, []float64{40})[0]
				neg := op.Forward([]int{1}, []float64{-40})[0]

				So(pos, ShouldBeGreaterThan, 1-1e-9)
				So(neg, ShouldBeLessThan, 1e-9)
			})

			Convey("It should map ±Inf to the logistic limits", func() {
				outPos := op.Forward([]int{1}, []float64{math.Inf(1)})
				outNeg := op.Forward([]int{1}, []float64{math.Inf(-1)})

				So(outPos[0], ShouldAlmostEqual, 1, 1e-12)
				So(outNeg[0], ShouldAlmostEqual, 0, 1e-12)
			})

			Convey("It should propagate NaN", func() {
				out := op.Forward([]int{1}, []float64{math.NaN()})

				So(math.IsNaN(out[0]), ShouldBeTrue)
			})
		})
	})
}

func benchmarkSigmoidForward(benchmark *testing.B, size int) {
	op := NewSigmoid()
	input := make([]float64, size)

	for index := range input {
		input[index] = float64(index%512)/256 - 1
	}

	shape := []int{len(input)}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		op.Forward(shape, input)
	}
}

func BenchmarkSigmoid_Forward(benchmark *testing.B) {
	// 4097 exercises SIMD prefixes plus a non-vector-aligned remainder (AVX2×4 / SSE2×2).
	benchmarkSigmoidForward(benchmark, 4097)
}

func BenchmarkSigmoid_Forward_Small(benchmark *testing.B) {
	benchmarkSigmoidForward(benchmark, 16)
}

func BenchmarkSigmoid_Forward_PowerOf2(benchmark *testing.B) {
	benchmarkSigmoidForward(benchmark, 4096)
}

func BenchmarkSigmoid_Forward_64(benchmark *testing.B) {
	benchmarkSigmoidForward(benchmark, 64)
}

func BenchmarkSigmoid_Forward_256(benchmark *testing.B) {
	benchmarkSigmoidForward(benchmark, 256)
}

func BenchmarkSigmoid_Forward_1024(benchmark *testing.B) {
	benchmarkSigmoidForward(benchmark, 1024)
}

func BenchmarkSigmoid_Forward_Large(benchmark *testing.B) {
	benchmarkSigmoidForward(benchmark, 1_048_576)
}
