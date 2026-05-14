package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const swigluTol = 5e-2

var benchSinkSwiGLU []float64

func referenceSwiGLU(input []float64) []float64 {
	n := len(input) / 2
	out := make([]float64, n)

	for index := 0; index < n; index++ {
		gate := input[index]
		value := input[n+index]
		swish := gate / (1 + math.Exp(-gate))
		out[index] = swish * value
	}

	return out
}

func TestSwiGLU(t *testing.T) {
	Convey("Given a SwiGLU operation", t, func() {
		op := NewSwiGLU()

		Convey("It should compute swish(gate) * value lane-wise", func() {
			gates := []float64{-2, -1, -0.5, 0, 0.5, 1, 1.5, 2}
			values := []float64{0.7, -1.3, 2.1, 0.0, -0.4, 1.5, -2.2, 0.9}
			input := append(append([]float64{}, gates...), values...)
			n := len(gates)
			out := forwardActivation(op, input)
			expected := referenceSwiGLU(input)

			for index := 0; index < n; index++ {
				So(out[index], ShouldAlmostEqual, expected[index], swigluTol)
			}
		})

		Convey("It should exercise the SIMD path on multiples of 4", func() {
			n := 16
			input := make([]float64, 2*n)

			for index := 0; index < 2*n; index++ {
				input[index] = float64(index%5)/2 - 1
			}

			out := forwardActivation(op, input)
			expected := referenceSwiGLU(input)

			for index := 0; index < n; index++ {
				So(out[index], ShouldAlmostEqual, expected[index], swigluTol)
			}
		})

		Convey("It should exercise the SSE2 path on even non-mult-of-4 lengths", func() {
			n := 6
			input := make([]float64, 2*n)

			for index := 0; index < 2*n; index++ {
				input[index] = float64(index%5)/2 - 1
			}

			out := forwardActivation(op, input)
			expected := referenceSwiGLU(input)

			for index := 0; index < n; index++ {
				So(out[index], ShouldAlmostEqual, expected[index], swigluTol)
			}
		})

		Convey("It should exercise the scalar tail on odd lengths", func() {
			n := 5
			input := make([]float64, 2*n)

			for index := 0; index < 2*n; index++ {
				input[index] = float64(index%5)/2 - 1
			}

			out := forwardActivation(op, input)
			expected := referenceSwiGLU(input)

			for index := 0; index < n; index++ {
				So(out[index], ShouldAlmostEqual, expected[index], swigluTol)
			}
		})

		Convey("It should propagate NaN gates", func() {
			input := []float64{math.NaN(), 1.0}
			out := forwardActivation(op, input)

			So(math.IsNaN(out[0]), ShouldBeTrue)
		})
	})
}

func BenchmarkSwiGLU_Forward(benchmark *testing.B) {
	op := NewSwiGLU()
	n := 4096
	input := make([]float64, 2*n)

	for index := range input {
		input[index] = float64(index%512)/256 - 1
	}

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{n}).
			WithInput(input)
		outputState, _ := op.Forward(stateDict)
		benchSinkSwiGLU = outputState.Out
	}

	_ = benchSinkSwiGLU
}
