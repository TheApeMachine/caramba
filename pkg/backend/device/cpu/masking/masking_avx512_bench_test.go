//go:build amd64

package masking

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkApplyMaskF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	length := 8192
	input := randomMaskingFloat32(length, 0x1840)
	mask := randomMaskingFloat32(length, 0x1841)
	output := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		ApplyMaskF32AVX512(&input[0], &mask[0], &output[0], length)
	}
}

func BenchmarkCausalMaskF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	side := 128
	output := make([]float32, side*side)

	b.SetBytes(int64(side * side * 4))
	b.ResetTimer()

	for b.Loop() {
		CausalMaskF32AVX512(&output[0], side, side)
	}
}

func BenchmarkALiBiBiasF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	side := 128
	scores := randomMaskingScores(side, side, 0x1842)
	slope := []float32{0.125}
	output := make([]float32, side*side)

	b.SetBytes(int64(side * side * 4))
	b.ResetTimer()

	for b.Loop() {
		ALiBiBiasF32AVX512(&scores[0], &slope[0], &output[0], side, side)
	}
}
