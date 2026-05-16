package activation

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const swigluReferenceTolerance = 1e-10

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

func referenceSwiGLURows(input []float64, rowWidth int) []float64 {
	rows := len(input) / rowWidth
	outputWidth := rowWidth / 2
	out := make([]float64, rows*outputWidth)

	for rowIndex := range rows {
		inputOffset := rowIndex * rowWidth
		outputOffset := rowIndex * outputWidth
		row := input[inputOffset : inputOffset+rowWidth]
		copy(out[outputOffset:outputOffset+outputWidth], referenceSwiGLU(row))
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
				So(out[index], ShouldAlmostEqual, expected[index], swigluReferenceTolerance)
			}
		})

		Convey("It should split gates and values inside each row", func() {
			input := []float64{
				1, 2, 3, 10, 20, 30,
				4, 5, 6, 40, 50, 60,
			}
			stateDict := state.NewDict().
				WithShape([]int{2, 6}).
				WithInput(input)

			outputState, err := op.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldHaveLength, 6)

			expected := referenceSwiGLURows(input, 6)

			for index := range expected {
				So(outputState.Out[index], ShouldAlmostEqual, expected[index], swigluReferenceTolerance)
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
				So(out[index], ShouldAlmostEqual, expected[index], swigluReferenceTolerance)
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
				So(out[index], ShouldAlmostEqual, expected[index], swigluReferenceTolerance)
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
				So(out[index], ShouldAlmostEqual, expected[index], swigluReferenceTolerance)
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
			WithShape([]int{len(input)}).
			WithInput(input)
		outputState, _ := op.Forward(stateDict)
		benchSinkSwiGLU = outputState.Out
	}

	_ = benchSinkSwiGLU
}
