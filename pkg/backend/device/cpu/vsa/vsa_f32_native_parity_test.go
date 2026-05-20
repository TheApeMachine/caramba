package vsa

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestVsaBindFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given VsaBindFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaBindFloat32Scalar for N=%d", length), func() {
				left, right := randomVSAVectors(length, 0xC301+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				VsaBindFloat32Native(got, left, right)
				VsaBindFloat32Scalar(want, left, right)

				assertVSASliceParity(t, got, want)
			})
		}
	})
}

func TestVsaBundleFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given VsaBundleFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaBundleFloat32Scalar for N=%d", length), func() {
				left, right := randomVSAVectors(length, 0xC302+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				VsaBundleFloat32Native(got, left, right)
				VsaBundleFloat32Scalar(want, left, right)

				assertVSASliceParity(t, got, want)
			})
		}
	})
}

func TestVsaPermuteFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given VsaPermuteFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaPermuteFloat32Scalar for N=%d", length), func() {
				input, _ := randomVSAVectors(length, 0xC303+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)
				shift := vsaParityShift(length)

				VsaPermuteFloat32Native(got, input, shift)
				VsaPermuteFloat32Scalar(want, input, shift)

				assertVSASliceParity(t, got, want)
			})
		}
	})
}

func TestVsaSimilarityFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given VsaSimilarityFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match VsaSimilarityFloat32Scalar for N=%d", length), func() {
				left, right := randomVSAVectors(length, 0xC304+int64(length))

				got := VsaSimilarityFloat32Native(left, right)
				want := VsaSimilarityFloat32Scalar(left, right)

				assertVSASimilarityParity(t, got, want)
			})
		}
	})
}
