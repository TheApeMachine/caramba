package vsa

import "testing"

func BenchmarkVsaBindFloat32Scalar(b *testing.B) {
	const benchLen = 8192
	left, right := randomVSAVectors(benchLen, 0xCC01)
	output := make([]float32, benchLen)

	for b.Loop() {
		VsaBindFloat32Scalar(output, left, right)
	}
}

func BenchmarkVsaBundleFloat32Scalar(b *testing.B) {
	const benchLen = 8192
	left, right := randomVSAVectors(benchLen, 0xCC02)
	output := make([]float32, benchLen)

	for b.Loop() {
		VsaBundleFloat32Scalar(output, left, right)
	}
}

func BenchmarkVsaPermuteFloat32Scalar(b *testing.B) {
	const benchLen = 8192
	input, _ := randomVSAVectors(benchLen, 0xCC03)
	output := make([]float32, benchLen)

	for b.Loop() {
		VsaPermuteFloat32Scalar(output, input, 17)
	}
}

func BenchmarkVsaSimilarityFloat32Scalar(b *testing.B) {
	const benchLen = 8192
	left, right := randomVSAVectors(benchLen, 0xCC04)

	for b.Loop() {
		_ = VsaSimilarityFloat32Scalar(left, right)
	}
}
