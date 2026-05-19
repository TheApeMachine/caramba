//go:build amd64

package reduction

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkSumF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	values := randomReductionFloat32Slice(8192, 1)
	length := len(values)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = SumF32AVX512(&values[0], length)
	}
}

func BenchmarkSumF32Generic(b *testing.B) {
	values := randomReductionFloat32Slice(8192, 1)
	length := len(values)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = SumF32Generic(&values[0], length)
	}
}

func BenchmarkMaxF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	values := randomReductionFloat32Slice(8192, 2)
	length := len(values)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = MaxF32AVX512(&values[0], length)
	}
}

func BenchmarkMaxF32Generic(b *testing.B) {
	values := randomReductionFloat32Slice(8192, 2)
	length := len(values)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = MaxF32Generic(&values[0], length)
	}
}
