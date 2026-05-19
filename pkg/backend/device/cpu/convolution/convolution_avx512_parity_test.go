//go:build amd64

package convolution

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512ConvolutionAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomConvolutionFloat32Pair(length int, seed int64) ([]float32, []float32) {
	rng := rand.New(rand.NewSource(seed))
	weight := make([]float32, length)
	patch := make([]float32, length)

	for index := range weight {
		weight[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
		patch[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return weight, patch
}

func assertConvPatchDotParity(
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

func TestConvPatchDotF32AVX512Parity(t *testing.T) {
	if !avx512ConvolutionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ConvPatchDotF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ConvPatchDotScalar for N=%d", length), func() {
				weight, patch := randomConvolutionFloat32Pair(length, 0xC0A+int64(length))

				want := ConvPatchDotScalar(weight, patch, length)
				got := ConvPatchDotF32AVX512(&weight[0], &patch[0], length)

				assertConvPatchDotParity(t, got, want, length)
			})
		}

		convey.Convey("It should match ConvPatchDotScalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				weight, patch := randomConvolutionFloat32Pair(length, 0xC0B+int64(length))

				want := ConvPatchDotScalar(weight, patch, length)
				got := ConvPatchDotFloat32AVX512Asm(&weight[0], &patch[0], length)

				assertConvPatchDotParity(t, got, want, length)
			}
		})
	})
}
