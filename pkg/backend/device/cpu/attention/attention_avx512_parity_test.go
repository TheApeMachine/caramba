//go:build amd64

package attention

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const flashAttentionAVX512MaxULP = 0

func avx512AttentionAvailable() bool {
	return cpu.X86.HasAVX512F
}

func flashOnlineUpdateScalar(
	acc, valueRow []float32,
	alpha, shifted float32,
	length int,
) {
	for index := range length {
		acc[index] = acc[index]*alpha + valueRow[index]*shifted
	}
}

func flashScaleScalar(
	out, acc []float32,
	invNormalizer float32,
	length int,
) {
	for index := range length {
		out[index] = acc[index] * invNormalizer
	}
}

func randomAttentionFloat32Slice(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return values
}

func TestFlashAttentionOnlineUpdateAVX512Parity(t *testing.T) {
	if !avx512AttentionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given FlashAttentionOnlineUpdateAVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				acc := randomAttentionFloat32Slice(length, 0xA70+int64(length))
				valueRow := randomAttentionFloat32Slice(length, 0xA71+int64(length))
				wantAcc := make([]float32, length)
				gotAcc := make([]float32, length)

				copy(wantAcc, acc)
				copy(gotAcc, acc)

				alpha := float32(0.37)
				shifted := float32(0.91)

				flashOnlineUpdateScalar(wantAcc, valueRow, alpha, shifted, length)
				flashAttentionOnlineUpdateAVX512(
					&gotAcc[0], &valueRow[0], alpha, shifted, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, gotAcc[:length], wantAcc[:length], flashAttentionAVX512MaxULP,
				)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				acc := randomAttentionFloat32Slice(length, 0xA72+int64(length))
				valueRow := randomAttentionFloat32Slice(length, 0xA73+int64(length))
				wantAcc := make([]float32, length)
				gotAcc := make([]float32, length)

				copy(wantAcc, acc)
				copy(gotAcc, acc)

				alpha := float32(0.41)
				shifted := float32(0.83)

				flashOnlineUpdateScalar(wantAcc, valueRow, alpha, shifted, length)
				FlashAttentionOnlineUpdateAVX512Asm(
					&gotAcc[0], &valueRow[0], alpha, shifted, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, gotAcc[:length], wantAcc[:length], flashAttentionAVX512MaxULP,
				)
			}
		})
	})
}

func TestFlashAttentionScaleAVX512Parity(t *testing.T) {
	if !avx512AttentionAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given FlashAttentionScaleAVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				acc := randomAttentionFloat32Slice(length, 0xA74+int64(length))
				wantOut := make([]float32, length)
				gotOut := make([]float32, length)
				invNormalizer := float32(0.125)

				flashScaleScalar(wantOut, acc, invNormalizer, length)
				flashAttentionScaleAVX512(&gotOut[0], &acc[0], invNormalizer, length)

				parity.AssertFloat32SlicesWithinULP(
					t, gotOut[:length], wantOut[:length], flashAttentionAVX512MaxULP,
				)
			})
		}

		convey.Convey("It should match scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				acc := randomAttentionFloat32Slice(length, 0xA75+int64(length))
				wantOut := make([]float32, length)
				gotOut := make([]float32, length)
				invNormalizer := float32(0.25)

				flashScaleScalar(wantOut, acc, invNormalizer, length)
				FlashAttentionScaleAVX512Asm(&gotOut[0], &acc[0], invNormalizer, length)

				parity.AssertFloat32SlicesWithinULP(
					t, gotOut[:length], wantOut[:length], flashAttentionAVX512MaxULP,
				)
			}
		})
	})
}
