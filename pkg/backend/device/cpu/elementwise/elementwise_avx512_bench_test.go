//go:build amd64

package elementwise

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkAddF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	left := make([]float32, 8192)
	right := make([]float32, 8192)
	out := make([]float32, 8192)
	length := len(out)

	b.ResetTimer()
	for b.Loop() {
		AddF32AVX512(&out[0], &left[0], &right[0], length)
	}
}

func BenchmarkMulF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	left := make([]float32, 8192)
	right := make([]float32, 8192)
	out := make([]float32, 8192)
	length := len(out)

	b.ResetTimer()
	for b.Loop() {
		MulF32AVX512(&out[0], &left[0], &right[0], length)
	}
}

func BenchmarkReluF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	source := make([]float32, 8192)
	out := make([]float32, 8192)
	length := len(out)

	b.ResetTimer()
	for b.Loop() {
		ReluF32AVX512(&out[0], &source[0], length)
	}
}

func BenchmarkAxpyF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	y := make([]float32, 8192)
	x := make([]float32, 8192)
	length := len(y)
	alpha := float32(0.5)

	b.ResetTimer()
	for b.Loop() {
		AxpyF32AVX512(&y[0], &x[0], alpha, length)
	}
}
