//go:build amd64

package dropout

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512DropoutAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomDropoutFloat32Slice(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	slice := make([]float32, length)

	for index := range slice {
		slice[index] = float32((rng.Float64() - 0.5) * 4.0)
	}

	return slice
}

func TestDropoutF32AVX512Parity(t *testing.T) {
	if !avx512DropoutAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given DropoutF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match DropoutF32Generic for N=%d", length), func() {
				source := randomDropoutFloat32Slice(length, 0xD08+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)
				seedGot := DropoutSeedState(0xC0FFEE)
				seedWant := seedGot
				keepProb := float32(0.75)

				DropoutF32AVX512(&got[0], &source[0], length, &seedGot, keepProb)
				DropoutF32Generic(&want[0], &source[0], length, &seedWant, keepProb)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match DropoutF32Generic via direct DropoutFloat32AVX512Asm", func() {
			for _, length := range parity.Lengths {
				convey.Convey(fmt.Sprintf("direct asm N=%d", length), func() {
					source := randomDropoutFloat32Slice(length, 0xA5EED+int64(length))
					got := make([]float32, length)
					want := make([]float32, length)
					seedGot := DropoutSeedState(0xFEEDFACE)
					seedWant := seedGot
					keepProb := float32(0.75)
					scale := float32(1.0 / keepProb)
					threshold := dropoutThreshold(keepProb)

					DropoutFloat32AVX512Asm(
						&got[0], &source[0], length,
						&seedGot[0], scale, threshold,
					)
					DropoutF32Generic(&want[0], &source[0], length, &seedWant, keepProb)

					parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
				})
			}
		})
	})
}
