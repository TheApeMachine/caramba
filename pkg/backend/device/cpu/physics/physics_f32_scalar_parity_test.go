package physics

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const physicsScalarMaxULP = 0

func TestLaplacianFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given LaplacianFloat32Scalar 1-D", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2400+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)
				dims := []int{length}
				invH2 := physicsInvH2ForTest()

				LaplacianFloat32Scalar(input, first, nil, dims, invH2)
				LaplacianFloat32Scalar(input, second, nil, dims, invH2)

				parity.AssertFloat32SlicesWithinULP(t, first, second, physicsScalarMaxULP)
			})
		}
	})
}

func TestLaplacian4Float32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given Laplacian4Float32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2410+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)

				Laplacian4Float32Scalar(input, first, physicsInvDenForTest())
				Laplacian4Float32Scalar(input, second, physicsInvDenForTest())

				parity.AssertFloat32SlicesWithinULP(t, first, second, physicsScalarMaxULP)
			})
		}
	})
}

func TestGrad1DFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given Grad1DFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2420+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)

				Grad1DFloat32Scalar(input, first, physicsInvTwoDxForTest())
				Grad1DFloat32Scalar(input, second, physicsInvTwoDxForTest())

				parity.AssertFloat32SlicesWithinULP(t, first, second, physicsScalarMaxULP)
			})
		}
	})
}

func TestQuantumPotentialFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given QuantumPotentialFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				density := randomPhysicsDensity(length, 0x2430+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)

				QuantumPotentialFloat32Scalar(density, first, physicsInvH2ForTest(), physicsQuantumScaleForTest())
				QuantumPotentialFloat32Scalar(density, second, physicsInvH2ForTest(), physicsQuantumScaleForTest())

				parity.AssertFloat32SlicesWithinULP(t, first, second, physicsScalarMaxULP)
			})
		}
	})
}

func TestMadelungContinuityFloat32ScalarParityLengths(t *testing.T) {
	convey.Convey("Given MadelungContinuityFloat32Scalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for N=%d", length), func() {
				density := randomPhysicsDensity(length, 0x2440+int64(length))
				velocity := randomPhysicsFloat32(length, 0x2441+int64(length))
				first := make([]float32, length)
				second := make([]float32, length)

				MadelungContinuityFloat32Scalar(density, velocity, first, physicsInvTwoDxForTest())
				MadelungContinuityFloat32Scalar(density, velocity, second, physicsInvTwoDxForTest())

				parity.AssertFloat32SlicesWithinULP(t, first, second, physicsScalarMaxULP)
			})
		}
	})
}
