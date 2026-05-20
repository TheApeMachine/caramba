package active_inference

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestFreeEnergyFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given FreeEnergyFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match FreeEnergyFloat32Scalar for N=%d", length), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA300+int64(length))

				got := FreeEnergyFloat32Native(likelihood, posterior, prior)
				want := FreeEnergyFloat32Scalar(likelihood, posterior, prior)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestExpectedFreeEnergyFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given ExpectedFreeEnergyFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ExpectedFreeEnergyFloat32Scalar for N=%d", length), func() {
				predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(
					length, 0xA310+int64(length),
				)

				got := ExpectedFreeEnergyFloat32Native(predictedObs, preferredObs, predictedState)
				want := ExpectedFreeEnergyFloat32Scalar(predictedObs, preferredObs, predictedState)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestBeliefUpdateFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given BeliefUpdateFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match BeliefUpdateFloat32Scalar for N=%d", length), func() {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA320+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				BeliefUpdateFloat32Native(likelihood, prior, got)
				BeliefUpdateFloat32Scalar(likelihood, prior, want)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}
	})
}

func TestPrecisionWeightFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given PrecisionWeightFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PrecisionWeightFloat32Scalar for N=%d", length), func() {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA330+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				PrecisionWeightFloat32Native(errors, precision, got)
				PrecisionWeightFloat32Scalar(errors, precision, want)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}
	})
}
