//go:build amd64

package rope

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const ropeAVX512MaxULP = 0

func avx512RopeAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestRopePairsF32AVX512Parity(t *testing.T) {
	if !avx512RopeAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given RopePairs float32 AVX-512", t, func() {
		for _, pairCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for pairs=%d", pairCount), func() {
				in, cos, sin := randomRopePairBuffers(pairCount, 0x5200+int64(pairCount))
				want := make([]float32, 2*pairCount)
				got := make([]float32, 2*pairCount)

				RopePairsGeneric(want, in, cos, sin)
				ropePairsF32AVX512(got, in, cos, sin)

				parity.AssertFloat32SlicesWithinULP(t, got, want, ropeAVX512MaxULP)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, pairCount := range parity.Lengths {
				in, cos, sin := randomRopePairBuffers(pairCount, 0x5201+int64(pairCount))
				want := make([]float32, 2*pairCount)
				got := make([]float32, 2*pairCount)

				RopePairsGeneric(want, in, cos, sin)
				RopePairsFloat32AVX512Asm(
					&got[0], &in[0], &cos[0], &sin[0], pairCount,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, ropeAVX512MaxULP)
			}
		})
	})
}
