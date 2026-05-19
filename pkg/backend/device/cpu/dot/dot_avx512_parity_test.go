//go:build amd64

package dot

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512DotAvailable() bool {
	return cpu.X86.HasAVX512F
}

func assertDotF32Parity(
	testingTB *testing.T,
	got, want float32,
	length int,
) {
	testingTB.Helper()

	tolerance := math.Max(math.Abs(float64(want)), 1.0) * float64(length) * 0x1p-50

	if math.Abs(float64(got-want)) > tolerance {
		testingTB.Fatalf(
			"N=%d got=%g want=%g diff=%g tol=%g",
			length, got, want, got-want, tolerance,
		)
	}
}

func TestDotF32AVX512Parity(t *testing.T) {
	if !avx512DotAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given DotF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match DotF32Generic for N=%d", length), func() {
				left := randomFloat32Slice(length, 0xD07+int64(length))
				right := randomFloat32Slice(length, 0xB17+int64(length))

				want := DotF32Generic(&left[0], &right[0], length)
				got := DotF32AVX512(&left[0], &right[0], length)

				assertDotF32Parity(t, got, want, length)
			})
		}
	})
}
