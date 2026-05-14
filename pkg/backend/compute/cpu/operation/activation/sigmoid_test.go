package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSigmoid(t *testing.T) {
	Convey("Given a Sigmoid operation", t, func() {
		op := NewSigmoid()

		Convey("Forward", func() {
			Convey(
				"When length is not a multiple of SIMD width (amd64: SSE2×2 / AVX2×4; arm64 NEON×2) scalar tail handling runs after any vector prefix",
				func() {
					out := forwardActivation(op, []float64{-1, 0, 1})

					So(out[0], ShouldBeLessThan, out[1])
					So(out[1], ShouldAlmostEqual, 0.5, 1e-9)
					So(out[2], ShouldBeGreaterThan, out[1])
				},
			)

			Convey("It should treat even-length vectors consistently", func() {
				out := forwardActivation(op, []float64{-2, -1, 1, 2})

				So(out[0], ShouldBeLessThan, out[1])
				So(out[1], ShouldBeLessThan, 0.5)
				So(out[2], ShouldBeGreaterThan, 0.5)
				So(out[3], ShouldBeGreaterThan, out[2])
			})

			Convey("It should return an empty slice for empty input", func() {
				So(forwardActivation(op, []float64{}), ShouldBeEmpty)
			})

			Convey("It should handle a single element", func() {
				So(forwardActivation(op, []float64{0})[0], ShouldAlmostEqual, 0.5, 1e-9)
			})

			Convey("It should approach 0 and 1 for large magnitude inputs", func() {
				pos := forwardActivation(op, []float64{40})[0]
				neg := forwardActivation(op, []float64{-40})[0]

				So(pos, ShouldBeGreaterThan, 1-1e-9)
				So(neg, ShouldBeLessThan, 1e-9)
			})

			Convey("It should map ±Inf to the logistic limits", func() {
				outPos := forwardActivation(op, []float64{math.Inf(1)})
				outNeg := forwardActivation(op, []float64{math.Inf(-1)})

				So(outPos[0], ShouldAlmostEqual, 1, 1e-12)
				So(outNeg[0], ShouldAlmostEqual, 0, 1e-12)
			})

			Convey("It should propagate NaN", func() {
				out := forwardActivation(op, []float64{math.NaN()})

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

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{len(input)}).
			WithInput(input)
		_, _ = op.Forward(stateDict)
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
