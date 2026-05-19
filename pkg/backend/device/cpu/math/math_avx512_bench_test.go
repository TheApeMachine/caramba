//go:build amd64

package math

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func avx512MathBenchAvailable() bool {
	return cpu.X86.HasAVX512F
}

func BenchmarkInvSqrtDimScaleF32AVX512(b *testing.B) {
	if !avx512MathBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	input := randomMathFloat32(8192, 0x2220)
	output := make([]float32, 8192)

	b.ResetTimer()

	for b.Loop() {
		InvSqrtDimScaleF32AVX512(output, input, 64)
	}
}

func BenchmarkInvSqrtDimScaleFloat32AVX512Asm(b *testing.B) {
	if !avx512MathBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	input := randomMathFloat32(8192, 0x2221)
	output := make([]float32, 8192)
	scale := float32(1.0 / 8.0)

	b.ResetTimer()

	for b.Loop() {
		InvSqrtDimScaleFloat32AVX512Asm(&output[0], &input[0], scale, len(output))
	}
}

func BenchmarkLogSumExpRowF32AVX512(b *testing.B) {
	if !avx512MathBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	row := randomMathFloat32(8192, 0x2222)

	b.ResetTimer()

	for b.Loop() {
		_ = logSumExpRowF32AVX512(row)
	}
}

func BenchmarkOuterF32AVX512(b *testing.B) {
	if !avx512MathBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	left := randomMathFloat32(64, 0x2223)
	right := randomMathFloat32(128, 0x2224)
	output := make([]float32, len(left)*len(right))

	b.ResetTimer()

	for b.Loop() {
		OuterF32AVX512(left, right, output)
	}
}

func BenchmarkInvSqrtDimScaleGeneric(b *testing.B) {
	input := randomMathFloat32(8192, 0x2225)
	output := make([]float32, 8192)

	b.ResetTimer()

	for b.Loop() {
		InvSqrtDimScaleGeneric(output, input, 64)
	}
}
