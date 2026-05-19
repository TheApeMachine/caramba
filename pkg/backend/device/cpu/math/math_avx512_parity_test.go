//go:build amd64

package math

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const mathAVX512MaxULP = 2

func avx512MathAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestInvSqrtDimScaleF32AVX512Parity(t *testing.T) {
	if !avx512MathAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given InvSqrtDimScaleF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match InvSqrtDimScaleGeneric for N=%d", length), func() {
				input := randomMathFloat32(length, 0x2210+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				InvSqrtDimScaleF32AVX512(got, input, 64)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match InvSqrtDimScaleGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMathFloat32(length, 0x2211+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				scale := float32(1.0 / 8.0)
				InvSqrtDimScaleFloat32AVX512Asm(&got[0], &input[0], scale, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestLogSumExpRowF32AVX512Parity(t *testing.T) {
	if !avx512MathAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given logSumExpRowF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpRowGeneric for N=%d", length), func() {
				row := randomMathFloat32(length, 0x2212+int64(length))
				want := LogSumExpRowGeneric(row)
				got := logSumExpRowF32AVX512(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathAVX512MaxULP)
			})
		}

		convey.Convey("It should match LogSumExpRowGeneric via direct asm parts at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomMathFloat32(length, 0x2213+int64(length))
				want := LogSumExpRowGeneric(row)

				var maximum float32
				var expSum float32

				LogSumExpRowPartsFloat32AVX512Asm(&row[0], length, &maximum, &expSum)
				got := logSumExpRowF32AVX512(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathAVX512MaxULP)
			}
		})
	})
}

func TestLogSumExpF32AVX512Parity(t *testing.T) {
	if !avx512MathAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given LogSumExpF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpGeneric for N=%d", length), func() {
				cols := mathSquareSide(length)
				rows := length / cols

				if rows < 1 {
					rows = 1
				}

				input := randomMathFloat32(rows*cols, 0x2214+int64(length))
				want := make([]float32, rows)
				got := make([]float32, rows)

				LogSumExpGeneric(input, cols, want)
				LogSumExpF32AVX512(input, cols, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, mathAVX512MaxULP)
			})
		}
	})
}

func TestOuterF32AVX512Parity(t *testing.T) {
	if !avx512MathAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given OuterF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match OuterGeneric for N=%d", length), func() {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2215+int64(length))
				right := randomMathFloat32(rightLen, 0x2216+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterF32AVX512(left, right, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match OuterGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2217+int64(length))
				right := randomMathFloat32(rightLen, 0x2218+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterFloat32AVX512Asm(&got[0], &left[0], &right[0], leftLen, rightLen)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
