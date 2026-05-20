package active_inference

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func randomActiveInferenceVectors(length int, seed int64) ([]float32, []float32, []float32) {
	rng := rand.New(rand.NewSource(seed))
	first := make([]float32, length)
	second := make([]float32, length)
	third := make([]float32, length)

	for index := range first {
		first[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
		second[index] = float32(rng.Float64() * 0.5)
		third[index] = float32(rng.Float64() * 0.5)
	}

	return first, second, third
}

func TestFreeEnergyFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given FreeEnergyFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA100+int64(length))

				first := FreeEnergyFloat32Scalar(likelihood, posterior, prior)
				second := FreeEnergyFloat32Scalar(likelihood, posterior, prior)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{first}, []float32{second}, activeInferenceScalarMaxULP,
				)
			})
		}
	})
}

func TestExpectedFreeEnergyFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given ExpectedFreeEnergyFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(
					length, 0xA110+int64(length),
				)

				first := ExpectedFreeEnergyFloat32Scalar(predictedObs, preferredObs, predictedState)
				second := ExpectedFreeEnergyFloat32Scalar(predictedObs, preferredObs, predictedState)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{first}, []float32{second}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestBeliefUpdateFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given BeliefUpdateFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA120+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)

				BeliefUpdateFloat32Scalar(likelihood, prior, first)
				BeliefUpdateFloat32Scalar(likelihood, prior, second)

				parity.AssertFloat32SlicesWithinULP(t, first, second, activeInferenceScalarMaxULP)
			})
		}
	})
}

func TestPrecisionWeightFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given PrecisionWeightFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA130+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)

				PrecisionWeightFloat32Scalar(errors, precision, first)
				PrecisionWeightFloat32Scalar(errors, precision, second)

				parity.AssertFloat32SlicesWithinULP(t, first, second, activeInferenceScalarMaxULP)
			})
		}
	})
}
