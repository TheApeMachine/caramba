//go:build amd64

package physics

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func avx512PhysicsBenchAvailable() bool {
	return cpu.X86.HasAVX512F
}

func BenchmarkLaplacian1DStencilF32AVX512Asm(b *testing.B) {
	if !avx512PhysicsBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	length := 8192
	left := randomPhysicsFloat32(length, 0x2580)
	center := randomPhysicsFloat32(length, 0x2581)
	right := randomPhysicsFloat32(length, 0x2582)
	output := make([]float32, length)
	invH2 := physicsInvH2ForTest()

	b.ResetTimer()

	for b.Loop() {
		Laplacian1DStencilF32AVX512Asm(&output[0], &left[0], &center[0], &right[0], invH2, length)
	}
}

func BenchmarkGrad1DStencilF32AVX512Asm(b *testing.B) {
	if !avx512PhysicsBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	length := 8192
	left := randomPhysicsFloat32(length, 0x2590)
	right := randomPhysicsFloat32(length, 0x2591)
	output := make([]float32, length)
	invTwoDx := physicsInvTwoDxForTest()

	b.ResetTimer()

	for b.Loop() {
		Grad1DStencilF32AVX512Asm(&output[0], &left[0], &right[0], invTwoDx, length)
	}
}

func BenchmarkLaplacian4StencilF32AVX512Asm(b *testing.B) {
	if !avx512PhysicsBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	length := 8192
	um2 := randomPhysicsFloat32(length, 0x25a0)
	um1 := randomPhysicsFloat32(length, 0x25a1)
	u0 := randomPhysicsFloat32(length, 0x25a2)
	up1 := randomPhysicsFloat32(length, 0x25a3)
	up2 := randomPhysicsFloat32(length, 0x25a4)
	output := make([]float32, length)
	invDen := physicsInvDenForTest()

	b.ResetTimer()

	for b.Loop() {
		Laplacian4StencilF32AVX512Asm(
			&output[0], &um2[0], &um1[0], &u0[0], &up1[0], &up2[0],
			invDen, length,
		)
	}
}
