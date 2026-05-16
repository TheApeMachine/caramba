//go:build linux && cgo && cuda

package cuda

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuactivation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestCUDAActivation_SwiGLU(test *testing.T) {
	Convey("Given CUDA SwiGLU activation", test, func() {
		activation := New()
		input := []float64{
			-3, -1.5, -0.25, 0, 0.75, 2, 4,
			0.5, -1.25, 3, 7, -0.5, 1.75, -2,
		}
		expectedState, err := cpuactivation.NewSwiGLU().Forward(
			state.NewDict().WithShape([]int{len(input) / 2}).WithInput(input),
		)
		So(err, ShouldBeNil)

		Convey("It should match the CPU gate*sigmoid(gate)*value contract", func() {
			output, err := activation.SwiGLU(input)

			So(err, ShouldBeNil)
			for index, expected := range expectedState.Out {
				So(output[index], ShouldAlmostEqual, expected, 1e-12)
			}
		})
	})
}

func BenchmarkCUDAActivation_SwiGLU(benchmark *testing.B) {
	activation := New()
	input := make([]float64, 8192)

	for index := range input {
		input[index] = float64(index%257)/64 - 2
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := activation.SwiGLU(input); err != nil {
			benchmark.Fatal(err)
		}
	}
}
