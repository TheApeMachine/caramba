package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLeakyReLU(t *testing.T) {
	Convey("Given a LeakyReLU operation", t, func() {
		op := NewLeakyReLU()
		withAlpha := func(stateDict *state.Dict) { stateDict.WithAlpha(0.25) }

		Convey("Forward", func() {
			Convey("It should pass positive values through unchanged", func() {
				out := forwardActivation(op, []float64{1, 2, 3}, withAlpha)

				So(out, ShouldResemble, []float64{1, 2, 3})
			})

			Convey("It should scale negative values by alpha", func() {
				out := forwardActivation(op, []float64{-1, -2, -3}, withAlpha)

				So(out, ShouldResemble, []float64{-0.25, -0.5, -0.75})
			})

			Convey("It should map zeros to zeros", func() {
				out := forwardActivation(op, []float64{0, 0, 0}, withAlpha)

				So(out, ShouldResemble, []float64{0, 0, 0})
			})

			Convey("It should not produce NaN for large-magnitude inputs", func() {
				out := forwardActivation(
					op, []float64{1e100, -1e100, 1e-100, -1e-100}, withAlpha,
				)

				So(len(out), ShouldEqual, 4)

				for index := range out {
					So(math.IsNaN(out[index]), ShouldBeFalse)
				}
			})

			Convey("It should handle even-length SIMD-aligned prefixes", func() {
				So(forwardActivation(op, []float64{1, -2, 3, -4}, withAlpha), ShouldResemble,
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

				out := forwardActivation(op, input, withAlpha)

				So(out, ShouldResemble, expected)
			})

			Convey("It should reject missing input tensors", func() {
				_, err := op.Forward(state.NewDict())

				So(err, ShouldNotBeNil)
			})

			Convey(
				"It should use the raw state input length",
				func() {
					out := forwardActivation(op, []float64{1, 2, 3}, withAlpha)

					So(out, ShouldResemble, []float64{1, 2, 3})
				},
			)

			Convey(
				"When length is not a multiple of SIMD width it exercises scalar tail handling (amd64: SSE2/AVX2 process two/four lanes; leftover elements run scalarLeakyReLU)",
				func() {
					out := forwardActivation(op, []float64{-2, 0, 4}, withAlpha)

					So(out, ShouldResemble, []float64{-0.5, 0, 4})
				},
			)
		})
	})
}

func BenchmarkLeakyReLU_Forward(benchmark *testing.B) {
	// Alpha distinct from unit tests (0.25): benchmarks commonly use 0.01 as in typical defaults.
	op := NewLeakyReLU()

	// 4097 elements forces a non-vector-aligned remainder vs SSE2 (×2) and AVX2 (×4) widths,
	// so each iteration stresses SIMD prefixes plus scalar tail.
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512) - 256
	}

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{len(input)}).
			WithInput(input).
			WithAlpha(0.01)
		_, _ = op.Forward(stateDict)
	}
}
