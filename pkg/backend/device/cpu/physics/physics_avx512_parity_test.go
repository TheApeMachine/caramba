//go:build amd64

package physics

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const physicsAVX512MaxULP = 0

func avx512PhysicsAvailable() bool {
	return cpu.X86.HasAVX512F
}

func laplacian1DStencilReference(
	left, center, right []float32,
	invH2 float32,
	out []float32,
) {
	for index := range out {
		out[index] = (left[index] + right[index] - 2*center[index]) * invH2
	}
}

func grad1DStencilReference(
	left, right []float32,
	invTwoDx float32,
	out []float32,
) {
	for index := range out {
		out[index] = (right[index] - left[index]) * invTwoDx
	}
}

func laplacian4StencilReference(
	um2, um1, u0, up1, up2 []float32,
	invDen float32,
	out []float32,
) {
	for index := range out {
		out[index] = laplacian4StencilFloat32(
			um2[index], um1[index], u0[index], up1[index], up2[index], invDen,
		)
	}
}

func TestLaplacian1DStencilF32AVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Laplacian1DStencilF32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the scalar stencil for N=%d", length), func() {
				left := randomPhysicsFloat32(length, 0x2500+int64(length))
				center := randomPhysicsFloat32(length, 0x2501+int64(length))
				right := randomPhysicsFloat32(length, 0x2502+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invH2 := physicsInvH2ForTest()

				laplacian1DStencilReference(left, center, right, invH2, want)
				Laplacian1DStencilF32AVX512Asm(&got[0], &left[0], &center[0], &right[0], invH2, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestGrad1DStencilF32AVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Grad1DStencilF32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the scalar stencil for N=%d", length), func() {
				left := randomPhysicsFloat32(length, 0x2510+int64(length))
				right := randomPhysicsFloat32(length, 0x2511+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invTwoDx := physicsInvTwoDxForTest()

				grad1DStencilReference(left, right, invTwoDx, want)
				Grad1DStencilF32AVX512Asm(&got[0], &left[0], &right[0], invTwoDx, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestLaplacian4StencilF32AVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Laplacian4StencilF32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the scalar stencil for N=%d", length), func() {
				um2 := randomPhysicsFloat32(length, 0x2520+int64(length))
				um1 := randomPhysicsFloat32(length, 0x2521+int64(length))
				u0 := randomPhysicsFloat32(length, 0x2522+int64(length))
				up1 := randomPhysicsFloat32(length, 0x2523+int64(length))
				up2 := randomPhysicsFloat32(length, 0x2524+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invDen := physicsInvDenForTest()

				laplacian4StencilReference(um2, um1, u0, up1, up2, invDen, want)
				Laplacian4StencilF32AVX512Asm(
					&got[0], &um2[0], &um1[0], &u0[0], &up1[0], &up2[0],
					invDen, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestLaplacianFloat32NativeAVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given LaplacianFloat32Native 1-D", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LaplacianFloat32Scalar for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2530+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				dims := []int{length}
				invH2 := physicsInvH2ForTest()

				LaplacianFloat32Scalar(input, want, nil, dims, invH2)
				LaplacianFloat32Native(input, got, nil, dims, invH2)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestLaplacian4Float32NativeAVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Laplacian4Float32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Laplacian4Float32Scalar for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2540+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invDen := physicsInvDenForTest()

				Laplacian4Float32Scalar(input, want, invDen)
				Laplacian4Float32Native(input, got, invDen)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestGrad1DFloat32NativeAVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Grad1DFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Grad1DFloat32Scalar for N=%d", length), func() {
				input := randomPhysicsFloat32(length, 0x2550+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invTwoDx := physicsInvTwoDxForTest()

				Grad1DFloat32Scalar(input, want, invTwoDx)
				Grad1DFloat32Native(input, got, invTwoDx)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestQuantumPotentialFloat32NativeAVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given QuantumPotentialFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match QuantumPotentialFloat32Scalar for N=%d", length), func() {
				density := randomPhysicsDensity(length, 0x2560+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invH2 := physicsInvH2ForTest()
				scale := physicsQuantumScaleForTest()

				QuantumPotentialFloat32Scalar(density, want, invH2, scale)
				QuantumPotentialFloat32Native(density, got, invH2, scale)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}

func TestMadelungContinuityFloat32NativeAVX512Parity(t *testing.T) {
	if !avx512PhysicsAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given MadelungContinuityFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match MadelungContinuityFloat32Scalar for N=%d", length), func() {
				density := randomPhysicsDensity(length, 0x2570+int64(length))
				velocity := randomPhysicsFloat32(length, 0x2571+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				invTwoDx := physicsInvTwoDxForTest()

				MadelungContinuityFloat32Scalar(density, velocity, want, invTwoDx)
				MadelungContinuityFloat32Native(density, velocity, got, invTwoDx)

				parity.AssertFloat32SlicesWithinULP(t, got, want, physicsAVX512MaxULP)
			})
		}
	})
}
