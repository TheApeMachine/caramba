//go:build amd64

package losses

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkMseSumF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	predictions, targets := randomLossesFloat32Pair(8192, 1)
	length := len(predictions)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		_ = MseSumF32AVX512(&predictions[0], &targets[0], length)
	}
}

func BenchmarkMseSumF32Generic(b *testing.B) {
	predictions, targets := randomLossesFloat32Pair(8192, 1)
	length := len(predictions)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		_ = MseSumF32Generic(&predictions[0], &targets[0], length)
	}
}

func BenchmarkMaeSumF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	predictions, targets := randomLossesFloat32Pair(8192, 2)
	length := len(predictions)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		_ = MaeSumF32AVX512(&predictions[0], &targets[0], length)
	}
}

func BenchmarkMaeSumF32Generic(b *testing.B) {
	predictions, targets := randomLossesFloat32Pair(8192, 2)
	length := len(predictions)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		_ = MaeSumF32Generic(&predictions[0], &targets[0], length)
	}
}
