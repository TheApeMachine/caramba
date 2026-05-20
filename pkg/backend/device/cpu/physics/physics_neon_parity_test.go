//go:build arm64

package physics

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const physicsNEONMaxULP = 2

// Madelung composes elementwise mul and grad NEON kernels.
const physicsNEONCompositeMaxULP = 2

func TestLaplacianFloat32NEONParityLengths(t *testing.T) {
	convey.Convey("Given LaplacianFloat32Native 1-D", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LaplacianFloat32Scalar for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2450+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				dims := []int{length}
				invH2 := physicsInvH2ForTest()

				LaplacianFloat32Scalar(input, want, nil, dims, invH2)
				LaplacianFloat32Native(input, got, nil, dims, invH2)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsNEONMaxULP)
			})
		}
	})
}

func TestLaplacian4Float32NEONParityLengths(t *testing.T) {
	convey.Convey("Given Laplacian4Float32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Laplacian4Float32Scalar for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2460+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invDen := physicsInvDenForTest()

				Laplacian4Float32Scalar(input, want, invDen)
				Laplacian4Float32Native(input, got, invDen)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsNEONMaxULP)
			})
		}
	})
}

func TestGrad1DFloat32NEONParityLengths(t *testing.T) {
	convey.Convey("Given Grad1DFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Grad1DFloat32Scalar for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2470+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invTwoDx := physicsInvTwoDxForTest()

				Grad1DFloat32Scalar(input, want, invTwoDx)
				Grad1DFloat32Native(input, got, invTwoDx)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsNEONMaxULP)
			})
		}
	})
}

func TestQuantumPotentialFloat32NEONParityLengths(t *testing.T) {
	convey.Convey("Given QuantumPotentialFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match QuantumPotentialFloat32Scalar for N=%d", length), func() {
				density := randomPhysicsDensity(length, 0x2480+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invH2 := physicsInvH2ForTest()
				scale := physicsQuantumScaleForTest()

				QuantumPotentialFloat32Scalar(density, want, invH2, scale)
				QuantumPotentialFloat32Native(density, got, invH2, scale)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsNEONMaxULP)
			})
		}
	})
}

func TestMadelungContinuityFloat32NEONParityLengths(t *testing.T) {
	convey.Convey("Given MadelungContinuityFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MadelungContinuityFloat32Scalar for N=%d", length), func() {
				density := randomPhysicsDensity(length, 0x2490+int64(length))
				velocity := randomPhysicsFloat32(length, 0x2491+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invTwoDx := physicsInvTwoDxForTest()

				MadelungContinuityFloat32Scalar(density, velocity, want, invTwoDx)
				MadelungContinuityFloat32Native(density, velocity, got, invTwoDx)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsNEONCompositeMaxULP)
			})
		}
	})
}
