package shape

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const shapeF32MaxULP = 0

func TestCopyContiguousFloat32NativeParity(t *testing.T) {
	convey.Convey("Given CopyContiguousFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CopyContiguousGeneric for N=%d", length), func() {
				source := randomShapeFloat32(length, 0x1700+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousFloat32Native(got, source)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeF32MaxULP)
			})
		}
	})
}

func TestWhereFloat32NativeParity(t *testing.T) {
	convey.Convey("Given WhereFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WhereGeneric for N=%d", length), func() {
				positive := randomShapeFloat32(length, 0x1710+int64(length))
				negative := randomShapeFloat32(length, 0x1711+int64(length))
				mask := shapeMaskBytes(length, 0x1712+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereFloat32Native(got, positive, negative, mask)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeF32MaxULP)
			})
		}
	})
}

func TestMaskedFillFloat32NativeParity(t *testing.T) {
	convey.Convey("Given MaskedFillFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaskedFillGeneric for N=%d", length), func() {
				input := randomShapeFloat32(length, 0x1720+int64(length))
				mask := shapeMaskBytes(length, 0x1721+int64(length))
				fillValue := float32(2.375)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillFloat32Native(got, input, fillValue, mask)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeF32MaxULP)
			})
		}
	})
}
