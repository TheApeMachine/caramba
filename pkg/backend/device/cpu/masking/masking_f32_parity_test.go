package masking

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const maskingF32MaxULP = 0

func TestApplyMaskFloat32NativeParity(t *testing.T) {
	convey.Convey("Given ApplyMaskFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match applyMaskF32Generic for N=%d", length), func() {
				input := randomMaskingFloat32(length, 0x1800+int64(length))
				mask := randomMaskingFloat32(length, 0x1801+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				ApplyMaskFloat32Native(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&got[0]),
					length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingF32MaxULP)
			})
		}
	})
}

func TestCausalMaskFloat32NativeParity(t *testing.T) {
	convey.Convey("Given CausalMaskFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match causalMaskF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				CausalMaskFloat32Native(unsafe.Pointer(&got[0]), side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingF32MaxULP)
			})
		}
	})
}

func TestALiBiBiasFloat32NativeParity(t *testing.T) {
	convey.Convey("Given ALiBiBiasFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match alibiBiasF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, 0x1810+int64(length))
				slope := []float32{0.125}
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				alibiBiasF32Generic(
					unsafe.Pointer(&scores[0]),
					unsafe.Pointer(&slope[0]),
					unsafe.Pointer(&want[0]),
					side,
					side,
				)
				ALiBiBiasFloat32Native(
					unsafe.Pointer(&scores[0]),
					unsafe.Pointer(&slope[0]),
					unsafe.Pointer(&got[0]),
					side,
					side,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingF32MaxULP)
			})
		}
	})
}
