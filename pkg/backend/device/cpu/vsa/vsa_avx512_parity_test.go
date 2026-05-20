//go:build amd64

package vsa

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512VSAAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestVsaBindF32AVX512Parity(t *testing.T) {
	if !avx512VSAAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given VsaBindF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaBindFloat32Scalar for N=%d", length), func() {
				left, right := randomVSAVectors(length, 0xC311+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				VsaBindF32AVX512(&got[0], &left[0], &right[0], length)
				VsaBindFloat32Scalar(want, left, right)

				assertVSASliceParity(t, got, want)
			})
		}
	})
}

func TestVsaBundleF32AVX512Parity(t *testing.T) {
	if !avx512VSAAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given VsaBundleF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaBundleFloat32Scalar for N=%d", length), func() {
				left, right := randomVSAVectors(length, 0xC312+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				VsaBundleF32AVX512(&got[0], &left[0], &right[0], length)
				VsaBundleFloat32Scalar(want, left, right)

				assertVSASliceParity(t, got, want)
			})
		}
	})
}

func TestVsaPermuteCopyF32AVX512Parity(t *testing.T) {
	if !avx512VSAAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given VsaPermuteCopyF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match copy for N=%d", length), func() {
				input, _ := randomVSAVectors(length, 0xC313+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				VsaPermuteCopyF32AVX512(&got[0], &input[0], length)
				copy(want, input)

				assertVSASliceParity(t, got, want)
			})
		}
	})
}

func TestVsaSimilarityF32AVX512Parity(t *testing.T) {
	if !avx512VSAAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given VsaSimilarityF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaSimilarityFloat32Scalar for N=%d", length), func() {
				left, right := randomVSAVectors(length, 0xC314+int64(length))

				got := VsaSimilarityF32AVX512(&left[0], &right[0], length)
				want := VsaSimilarityFloat32Scalar(left, right)

				assertVSASimilarityParity(t, got, want)
			})
		}
	})
}
