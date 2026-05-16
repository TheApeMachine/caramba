package activation

import (
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSwish(t *testing.T) {
	Convey("Given a Swish operation", t, func() {
		op := NewSwish()

		Convey("Forward", func() {
			Convey("It should handle vector tails without dropping elements", func() {
				input := []float64{-2, -1, 0, 1, 2}
				out := forwardActivation(op, input)

				So(out, ShouldHaveLength, len(input))
				So(out[0], ShouldBeLessThan, 0)
				So(out[2], ShouldAlmostEqual, 0, 1e-12)
				So(out[4], ShouldBeGreaterThan, out[3])
			})

			Convey("It should match swish over the sampled range", func() {
				input := make([]float64, 0, 2401)

				for index := -1200; index <= 1200; index++ {
					input = append(input, float64(index)/100)
				}

				out := forwardActivation(op, input)
				maxError := 0.0

				for index, value := range input {
					expected := value / (1 + math.Exp(-value))
					errorValue := math.Abs(out[index] - expected)

					if errorValue > maxError {
						maxError = errorValue
					}
				}

				So(maxError, ShouldBeLessThanOrEqualTo, 1e-10)
			})

			Convey("It should reject a nil state dict", func() {
				_, err := op.Forward(nil)

				So(err, ShouldNotBeNil)
			})
		})
	})
}

func BenchmarkSwish_Forward(benchmark *testing.B) {
	op := NewSwish()
	input := make([]float64, 4097)

	for index := range input {
		input[index] = float64(index%512)/64 - 4
	}

	stateDict := state.NewDict().
		WithShape([]int{len(input)}).
		WithInput(input)

	for benchmark.Loop() {
		_, _ = op.Forward(stateDict)
	}
}
