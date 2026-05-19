//go:build amd64

package causal

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const causalAVX512MaxULP = 0

func avx512CausalAvailable() bool {
	return cpu.X86.HasAVX512F
}

func assertStridedDotF32Parity(
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

func TestCateF32AVX512Parity(t *testing.T) {
	if !avx512CausalAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given CateFloat32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match cateF32Generic for N=%d", length), func() {
				treated := randomCausalFloat32Slice(length, 0xCA30+int64(length))
				control := randomCausalFloat32Slice(length, 0xCA31+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				cateF32Generic(treated, control, want)
				CateFloat32AVX512Asm(&treated[0], &control[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, causalAVX512MaxULP)
			})
		}
	})
}

func TestCounterfactualF32AVX512Parity(t *testing.T) {
	if !avx512CausalAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given CounterfactualFloat32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match counterfactualF32Generic for N=%d", length), func() {
				observedY := randomCausalFloat32Slice(length, 0xCA32+int64(length))
				observedX := randomCausalFloat32Slice(length, 0xCA33+int64(length))
				counterfactualX := randomCausalFloat32Slice(length, 0xCA34+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				const slope = float32(-1.25)

				counterfactualF32Generic(want, observedY, observedX, counterfactualX, slope)
				CounterfactualFloat32AVX512Asm(
					&got[0], &observedY[0], &observedX[0], &counterfactualX[0],
					slope, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, causalAVX512MaxULP)
			})
		}
	})
}

func TestStridedDotF32AVX512Parity(t *testing.T) {
	if !avx512CausalAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given StridedDotFloat32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match stridedDotF32Generic for N=%d", length), func() {
				const stride = 7
				values := randomCausalFloat32Slice(length*stride, 0xCA35+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA36+int64(length))

				want := stridedDotF32Generic(values, stride, weights, length)
				got := StridedDotFloat32AVX512Asm(&values[0], stride, &weights[0], length)

				assertStridedDotF32Parity(t, got, want, length)
			})
		}

		convey.Convey("It should match stridedDotF32Generic at stride 1", func() {
			for _, length := range parity.Lengths {
				values := randomCausalFloat32Slice(length, 0xCA37+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA38+int64(length))

				want := stridedDotF32Generic(values, 1, weights, length)
				got := StridedDotFloat32AVX512Asm(&values[0], 1, &weights[0], length)

				assertStridedDotF32Parity(t, got, want, length)
			}
		})
	})
}
