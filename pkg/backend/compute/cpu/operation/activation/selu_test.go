package activation

import (
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSELU(test *testing.T) {
	Convey("Given a SELU operation", test, func() {
		operation := NewSELU()

		Convey("Forward", func() {
			Convey("It should match the self-normalizing SELU equation", func() {
				input := make([]float64, 0, 2401)

				for index := -1200; index <= 1200; index++ {
					input = append(input, float64(index)/100)
				}

				out := forwardActivation(operation, input)
				maxError := 0.0

				for index, value := range input {
					expected := seluScale * value

					if value <= 0 {
						expected = seluScaleAlpha * (math.Exp(value) - 1)
					}

					errorValue := math.Abs(out[index] - expected)

					if errorValue > maxError {
						maxError = errorValue
					}
				}

				So(maxError, ShouldBeLessThanOrEqualTo, 1e-9)
			})

			Convey("It should reject a nil state dict", func() {
				_, err := operation.Forward(nil)

				So(err, ShouldNotBeNil)
			})
		})
	})
}

func BenchmarkSELU_Forward(benchmark *testing.B) {
	operation := NewSELU()
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512)/64 - 4
	}

	stateDict := state.NewDict().
		WithShape([]int{len(input)}).
		WithInput(input)

	for benchmark.Loop() {
		_, _ = operation.Forward(stateDict)
	}
}
