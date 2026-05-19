//go:build amd64

package layernorm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const layernormAVX512MaxULP = 0

func avx512LayerNormAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomLayerNormRow(length int, seed int64) ([]float32, []float32, []float32) {
	rng := rand.New(rand.NewSource(seed))
	row := make([]float32, length)
	scale := make([]float32, length)
	bias := make([]float32, length)

	for index := range row {
		row[index] = float32((rng.Float64() - 0.5) * 4)
		scale[index] = float32(rng.Float64()*0.5 + 0.75)
		bias[index] = float32((rng.Float64() - 0.5) * 0.25)
	}

	return row, scale, bias
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

func TestLayerNormSquaredDiffSumF32AVX512Parity(t *testing.T) {
	if !avx512LayerNormAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given LayerNormSquaredDiffSum float32 AVX-512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, _, _ := randomLayerNormRow(length, 0x4E00+int64(length))
				mean := float32(0.125)

				want := LayerNormSquaredDiffSumGeneric(row, mean)
				got := layerNormSquaredDiffSumF32AVX512(row, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row, _, _ := randomLayerNormRow(length, 0x4E01+int64(length))
				mean := float32(-0.25)

				want := LayerNormSquaredDiffSumGeneric(row, mean)
				got := LayerNormSquaredDiffSumFloat32AVX512Asm(&row[0], length, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			}
		})
	})
}

func TestLayerNormApplyRowF32AVX512Parity(t *testing.T) {
	if !avx512LayerNormAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given LayerNormApplyRow float32 AVX-512", t, func() {
		mean := float32(0.05)
		invStdDev := float32(1.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, scale, bias := randomLayerNormRow(length, 0x4E02+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				LayerNormApplyRowGeneric(want, row, scale, bias, mean, invStdDev)
				layerNormApplyRowF32AVX512(got, row, scale, bias, mean, invStdDev)

				parity.AssertFloat32SlicesWithinULP(t, got, want, layernormAVX512MaxULP)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row, scale, bias := randomLayerNormRow(length, 0x4E03+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				LayerNormApplyRowGeneric(want, row, scale, bias, mean, invStdDev)
				LayerNormApplyRowFloat32AVX512Asm(
					&got[0], &row[0], &scale[0], &bias[0],
					length, mean, invStdDev,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, layernormAVX512MaxULP)
			}
		})
	})
}
