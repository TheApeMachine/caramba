//go:build amd64

package active_inference

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512ActiveInferenceAvailable() bool {
	return cpu.X86.HasAVX512F
}

func assertScalarSumParity(
	testingTB *testing.T,
	got, want float32,
	length int,
) {
	testingTB.Helper()

	tolerance := math.Max(math.Abs(float64(want)), 1.0) * float64(length) * 0x1p-50

	if math.Abs(float64(got-want)) > tolerance {
		testingTB.Fatalf(
			"N=%d got=%g want=%g diff=%g tol=%g",
			length, got, want, got-want, tolerance,
		)
	}
}

func TestFreeEnergyF32AVX512Parity(t *testing.T) {
	if !avx512ActiveInferenceAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given FreeEnergyF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match FreeEnergyF32Generic for N=%d", length), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA200+int64(length))

				want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], length)
				got := FreeEnergyF32AVX512(&likelihood[0], &posterior[0], &prior[0], length)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestExpectedFreeEnergyF32AVX512Parity(t *testing.T) {
	if !avx512ActiveInferenceAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ExpectedFreeEnergyF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ExpectedFreeEnergyF32Generic for N=%d", length), func() {
				predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(
					length, 0xA210+int64(length),
				)

				want := ExpectedFreeEnergyF32Generic(
					&predictedObs[0], &preferredObs[0], &predictedState[0],
					length, length,
				)
				got := ExpectedFreeEnergyF32AVX512(
					&predictedObs[0], &preferredObs[0], &predictedState[0],
					length, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestBeliefUpdateF32AVX512Parity(t *testing.T) {
	if !avx512ActiveInferenceAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given BeliefUpdateF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match BeliefUpdateF32Generic for N=%d", length), func() {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA220+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateF32AVX512(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match BeliefUpdateF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA221+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateFloat32AVX512Asm(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}

func TestPrecisionWeightF32AVX512Parity(t *testing.T) {
	if !avx512ActiveInferenceAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given PrecisionWeightF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PrecisionWeightF32Generic for N=%d", length), func() {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA230+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightF32AVX512(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match PrecisionWeightF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA231+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightFloat32AVX512Asm(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}

func TestFreeEnergyFloat32AVX512AsmBlockParity(t *testing.T) {
	if !avx512ActiveInferenceAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given FreeEnergyFloat32AVX512Asm on 8-wide blocks", t, func() {
		for _, length := range parity.Lengths {
			blockCount := length &^ 7

			if blockCount == 0 {
				continue
			}

			convey.Convey(fmt.Sprintf("It should match generic prefix for block N=%d", blockCount), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA240+int64(length))

				want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], blockCount)
				got := FreeEnergyFloat32AVX512Asm(&likelihood[0], &posterior[0], &prior[0], blockCount)

				assertScalarSumParity(t, got, want, blockCount)
			})
		}
	})
}
