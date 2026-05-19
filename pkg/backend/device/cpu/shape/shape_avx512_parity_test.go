//go:build amd64

package shape

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const shapeAVX512MaxULP = 0

func avx512ShapeAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestCopyContiguousF32AVX512Parity(t *testing.T) {
	if !avx512ShapeAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given CopyContiguousF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CopyContiguousGeneric for N=%d", length), func() {
				source := randomShapeFloat32(length, 0x1740+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousF32AVX512(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX512MaxULP)
			})
		}

		convey.Convey("It should match CopyContiguousGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomShapeFloat32(length, 0x1741+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				CopyContiguousGeneric(want, source)
				CopyContiguousFloat32AVX512Asm(&got[0], &source[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX512MaxULP)
			}
		})
	})
}

func TestWhereF32AVX512Parity(t *testing.T) {
	if !avx512ShapeAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given WhereF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WhereGeneric for N=%d", length), func() {
				positive := randomShapeFloat32(length, 0x1750+int64(length))
				negative := randomShapeFloat32(length, 0x1751+int64(length))
				mask := shapeMaskBytes(length, 0x1752+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereF32AVX512(&got[0], &positive[0], &negative[0], mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX512MaxULP)
			})
		}

		convey.Convey("It should match WhereGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				positive := randomShapeFloat32(length, 0x1753+int64(length))
				negative := randomShapeFloat32(length, 0x1754+int64(length))
				mask := shapeMaskBytes(length, 0x1755+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				WhereGeneric(want, positive, negative, mask)
				WhereFloat32AVX512Asm(&got[0], &positive[0], &negative[0], &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX512MaxULP)
			}
		})
	})
}

func TestMaskedFillF32AVX512Parity(t *testing.T) {
	if !avx512ShapeAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MaskedFillF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MaskedFillGeneric for N=%d", length), func() {
				input := randomShapeFloat32(length, 0x1760+int64(length))
				mask := shapeMaskBytes(length, 0x1761+int64(length))
				fillValue := float32(-0.75)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillF32AVX512(&got[0], &input[0], fillValue, mask, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX512MaxULP)
			})
		}

		convey.Convey("It should match MaskedFillGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomShapeFloat32(length, 0x1762+int64(length))
				mask := shapeMaskBytes(length, 0x1763+int64(length))
				fillValue := float32(3.5)
				want := make([]float32, length)
				got := make([]float32, length)

				MaskedFillGeneric(want, input, fillValue, mask)
				MaskedFillFloat32AVX512Asm(&got[0], &input[0], fillValue, &mask[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, shapeAVX512MaxULP)
			}
		})
	})
}
