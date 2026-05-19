package math

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestInvSqrtDimScaleGenericParityLengths(t *testing.T) {
	convey.Convey("Given InvSqrtDimScaleGeneric", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should scale N=%d elements", length), func() {
				input := randomMathFloat32(length, 0x2200+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				InvSqrtDimScaleGeneric(got, input, 64)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestLogSumExpRowGenericParityLengths(t *testing.T) {
	convey.Convey("Given LogSumExpRowGeneric", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should reduce one row of N=%d", length), func() {
				row := randomMathFloat32(length, 0x2201+int64(length))
				first := LogSumExpRowGeneric(row)
				second := LogSumExpRowGeneric(row)

				if first != second {
					t.Fatalf("N=%d non-deterministic got=%v second=%v", length, first, second)
				}
			})
		}
	})
}

func TestOuterGenericParityLengths(t *testing.T) {
	convey.Convey("Given OuterGeneric", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should form an outer product for N=%d", length), func() {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2202+int64(length))
				right := randomMathFloat32(rightLen, 0x2203+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterGeneric(left, right, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}
