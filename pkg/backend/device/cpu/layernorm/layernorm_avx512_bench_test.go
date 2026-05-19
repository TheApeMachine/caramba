//go:build amd64

package layernorm

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func avx512LayerNormBenchAvailable() bool {
	return cpu.X86.HasAVX512F
}

func BenchmarkLayerNormSquaredDiffSumF32AVX512(b *testing.B) {
	if !avx512LayerNormBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	const length = 1024
	row, _, _ := randomLayerNormRow(length, 0x4EB0)
	mean := float32(0.1)

	b.ResetTimer()

	for b.Loop() {
		_ = layerNormSquaredDiffSumF32AVX512(row, mean)
	}
}

func BenchmarkLayerNormSquaredDiffSumF32Generic(b *testing.B) {
	const length = 1024
	row, _, _ := randomLayerNormRow(length, 0x4EB1)
	mean := float32(0.1)

	b.ResetTimer()

	for b.Loop() {
		_ = LayerNormSquaredDiffSumGeneric(row, mean)
	}
}

func BenchmarkLayerNormApplyRowF32AVX512(b *testing.B) {
	if !avx512LayerNormBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	const length = 1024
	row, scale, bias := randomLayerNormRow(length, 0x4EB2)
	out := make([]float32, length)
	mean := float32(0.05)
	invStdDev := float32(1.25)

	b.ResetTimer()

	for b.Loop() {
		layerNormApplyRowF32AVX512(out, row, scale, bias, mean, invStdDev)
	}
}

func BenchmarkLayerNormApplyRowF32Generic(b *testing.B) {
	const length = 1024
	row, scale, bias := randomLayerNormRow(length, 0x4EB3)
	out := make([]float32, length)
	mean := float32(0.05)
	invStdDev := float32(1.25)

	b.ResetTimer()

	for b.Loop() {
		LayerNormApplyRowGeneric(out, row, scale, bias, mean, invStdDev)
	}
}
