//go:build arm64

package physics

import "testing"

func BenchmarkLaplacianFloat32Native1D(b *testing.B) {
	length := 8192
	input := randomPhysicsFloat32(length, 0x24a0)
	output := make([]float32, length)
	dims := []int{length}
	invH2 := physicsInvH2ForTest()

	b.ResetTimer()

	for b.Loop() {
		LaplacianFloat32Native(input, output, nil, dims, invH2)
	}
}

func BenchmarkLaplacianFloat32Scalar1D(b *testing.B) {
	length := 8192
	input := randomPhysicsFloat32(length, 0x24a1)
	output := make([]float32, length)
	dims := []int{length}
	invH2 := physicsInvH2ForTest()

	b.ResetTimer()

	for b.Loop() {
		LaplacianFloat32Scalar(input, output, nil, dims, invH2)
	}
}

func BenchmarkLaplacian4Float32Native(b *testing.B) {
	length := 8192
	input := randomPhysicsFloat32(length, 0x24a2)
	output := make([]float32, length)
	invDen := physicsInvDenForTest()

	b.ResetTimer()

	for b.Loop() {
		Laplacian4Float32Native(input, output, invDen)
	}
}

func BenchmarkGrad1DFloat32Native(b *testing.B) {
	length := 8192
	input := randomPhysicsFloat32(length, 0x24a3)
	output := make([]float32, length)
	invTwoDx := physicsInvTwoDxForTest()

	b.ResetTimer()

	for b.Loop() {
		Grad1DFloat32Native(input, output, invTwoDx)
	}
}
