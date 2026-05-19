//go:build amd64

package masking

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const maskingAVX512MaxULP = 0

func avx512MaskingAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestApplyMaskF32AVX512Parity(t *testing.T) {
	if !avx512MaskingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ApplyMaskF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match applyMaskF32Generic for N=%d", length), func() {
				input := randomMaskingFloat32(length, 0x1820+int64(length))
				mask := randomMaskingFloat32(length, 0x1821+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				ApplyMaskF32AVX512(&input[0], &mask[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingAVX512MaxULP)
			})
		}

		convey.Convey("It should match applyMaskF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMaskingFloat32(length, 0x1822+int64(length))
				mask := randomMaskingFloat32(length, 0x1823+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				ApplyMaskFloat32AVX512Asm(&input[0], &mask[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingAVX512MaxULP)
			}
		})
	})
}

func TestCausalMaskF32AVX512Parity(t *testing.T) {
	if !avx512MaskingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given CausalMaskF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match causalMaskF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				CausalMaskF32AVX512(&got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingAVX512MaxULP)
			})
		}

		convey.Convey("It should match causalMaskF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				CausalMaskFloat32AVX512Asm(&got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingAVX512MaxULP)
			}
		})
	})
}

func TestALiBiBiasF32AVX512Parity(t *testing.T) {
	if !avx512MaskingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given ALiBiBiasF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match alibiBiasF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, 0x1830+int64(length))
				slope := []float32{0.25}
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				alibiBiasF32Generic(
					unsafe.Pointer(&scores[0]),
					unsafe.Pointer(&slope[0]),
					unsafe.Pointer(&want[0]),
					side,
					side,
				)
				ALiBiBiasF32AVX512(&scores[0], &slope[0], &got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingAVX512MaxULP)
			})
		}

		convey.Convey("It should match alibiBiasF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, 0x1831+int64(length))
				slope := []float32{0.5}
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				alibiBiasF32Generic(
					unsafe.Pointer(&scores[0]),
					unsafe.Pointer(&slope[0]),
					unsafe.Pointer(&want[0]),
					side,
					side,
				)
				ALiBiBiasFloat32AVX512Asm(&scores[0], &slope[0], &got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingAVX512MaxULP)
			}
		})
	})
}
