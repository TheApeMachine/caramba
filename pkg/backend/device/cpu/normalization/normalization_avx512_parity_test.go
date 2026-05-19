//go:build amd64

package normalization

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const normalizationAVX512MaxULP = 0

func avx512NormalizationAvailable() bool {
	return cpu.X86.HasAVX512F
}

func assertSquaredDiffSumParity(
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

func TestNormSquaredDiffSumF32AVX512Parity(t *testing.T) {
	if !avx512NormalizationAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given NormSquaredDiffSum float32 AVX-512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row := randomNormalizationRow(length, 0x4F00+int64(length))
				mean := float32(0.125)

				want := NormSquaredDiffSumGeneric(row, mean)
				got := normSquaredDiffSumF32AVX512(row, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomNormalizationRow(length, 0x4F01+int64(length))
				mean := float32(-0.25)

				want := NormSquaredDiffSumGeneric(row, mean)
				got := NormSquaredDiffSumFloat32AVX512Asm(&row[0], length, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			}
		})
	})
}

func TestNormApplyConstScaleBiasF32AVX512Parity(t *testing.T) {
	if !avx512NormalizationAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given NormApplyConstScaleBias float32 AVX-512", t, func() {
		mean := float32(0.05)
		invStdDev := float32(1.25)
		scale := float32(0.9)
		bias := float32(-0.1)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row := randomNormalizationRow(length, 0x4F02+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				NormApplyConstScaleBiasGeneric(want, row, mean, invStdDev, scale, bias)
				normApplyConstScaleBiasF32AVX512(got, row, mean, invStdDev, scale, bias)

				parity.AssertFloat32SlicesWithinULP(t, got, want, normalizationAVX512MaxULP)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomNormalizationRow(length, 0x4F03+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				NormApplyConstScaleBiasGeneric(want, row, mean, invStdDev, scale, bias)
				NormApplyConstScaleBiasFloat32AVX512Asm(
					&got[0], &row[0], length,
					mean, invStdDev, scale, bias,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, normalizationAVX512MaxULP)
			}
		})
	})
}
