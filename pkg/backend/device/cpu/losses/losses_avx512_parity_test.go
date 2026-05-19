//go:build amd64

package losses

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512LossesAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomLossesFloat32Pair(length int, seed int64) ([]float32, []float32) {
	rng := rand.New(rand.NewSource(seed))
	predictions := make([]float32, length)
	targets := make([]float32, length)

	for index := range predictions {
		predictions[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
		targets[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return predictions, targets
}

func assertPairSumParity(
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

func TestMseSumF32AVX512Parity(t *testing.T) {
	if !avx512LossesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MseSumF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MseSumF32Generic for N=%d", length), func() {
				predictions, targets := randomLossesFloat32Pair(length, 0x5E0+int64(length))

				want := MseSumF32Generic(&predictions[0], &targets[0], length)
				got := MseSumF32AVX512(&predictions[0], &targets[0], length)

				assertPairSumParity(t, got, want, length)
			})
		}

		convey.Convey("It should match MseSumF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				predictions, targets := randomLossesFloat32Pair(length, 0x5E1+int64(length))

				want := MseSumF32Generic(&predictions[0], &targets[0], length)
				got := MseSumFloat32AVX512Asm(&predictions[0], &targets[0], length)

				assertPairSumParity(t, got, want, length)
			}
		})
	})
}

func TestMaeSumF32AVX512Parity(t *testing.T) {
	if !avx512LossesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MaeSumF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaeSumF32Generic for N=%d", length), func() {
				predictions, targets := randomLossesFloat32Pair(length, 0xAB0+int64(length))

				want := MaeSumF32Generic(&predictions[0], &targets[0], length)
				got := MaeSumF32AVX512(&predictions[0], &targets[0], length)

				assertPairSumParity(t, got, want, length)
			})
		}

		convey.Convey("It should match MaeSumF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				predictions, targets := randomLossesFloat32Pair(length, 0xAB1+int64(length))

				want := MaeSumF32Generic(&predictions[0], &targets[0], length)
				got := MaeSumFloat32AVX512Asm(&predictions[0], &targets[0], length)

				assertPairSumParity(t, got, want, length)
			}
		})
	})
}
