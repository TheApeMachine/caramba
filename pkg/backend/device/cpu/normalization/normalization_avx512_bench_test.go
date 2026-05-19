//go:build amd64

package normalization

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkNormSquaredDiffSumF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	row := randomNormalizationRow(8192, 1)
	mean := float32(0.1)
	length := len(row)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = normSquaredDiffSumF32AVX512(row, mean)
	}
}

func BenchmarkNormSquaredDiffSumF32Generic(b *testing.B) {
	row := randomNormalizationRow(8192, 1)
	mean := float32(0.1)
	length := len(row)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = NormSquaredDiffSumGeneric(row, mean)
	}
}

func BenchmarkNormApplyConstScaleBiasF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	row := randomNormalizationRow(8192, 2)
	out := make([]float32, len(row))
	length := len(row)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		normApplyConstScaleBiasF32AVX512(out, row, 0.05, 1.25, 0.9, -0.1)
	}
}

func BenchmarkNormApplyConstScaleBiasF32Generic(b *testing.B) {
	row := randomNormalizationRow(8192, 2)
	out := make([]float32, len(row))
	length := len(row)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		NormApplyConstScaleBiasGeneric(out, row, 0.05, 1.25, 0.9, -0.1)
	}
}
