//go:build amd64

package sampling

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const samplingAVX512MaxULP = 2

func avx512SamplingAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestGreedySampleF32AVX512Parity(t *testing.T) {
	if !avx512SamplingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given GreedySampleF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match GreedySampleGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x1600+int64(length))

				want := GreedySampleGeneric(logits)
				got := GreedySampleF32AVX512(&logits[0], length)

				if got != want {
					t.Fatalf("N=%d got=%d want=%d", length, got, want)
				}
			})
		}

		convey.Convey("It should match GreedySampleGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				logits := randomSamplingLogits(length, 0x1601+int64(length))

				want := GreedySampleGeneric(logits)
				got := GreedySampleFloat32AVX512Asm(&logits[0], length)

				if got != want {
					t.Fatalf("direct asm N=%d got=%d want=%d", length, got, want)
				}
			}
		})
	})
}

func TestSamplingSoftmaxRowF32AVX512Parity(t *testing.T) {
	if !avx512SamplingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given SamplingSoftmaxRowF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SamplingSoftmaxRowGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x1610+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(0.85)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowF32AVX512(&logits[0], &got[0], temperature, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingAVX512MaxULP)
			})
		}

		convey.Convey("It should match SamplingSoftmaxRowGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				logits := randomSamplingLogits(length, 0x1611+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(1.25)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowFloat32AVX512Asm(
					&logits[0], &got[0], temperature, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingAVX512MaxULP)
			}
		})
	})
}
