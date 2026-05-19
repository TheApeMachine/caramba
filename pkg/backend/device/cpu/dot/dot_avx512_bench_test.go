//go:build amd64

package dot

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkDotF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	left := randomFloat32Slice(8192, 1)
	right := randomFloat32Slice(8192, 2)
	length := len(left)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		_ = DotF32AVX512(&left[0], &right[0], length)
	}
}

func BenchmarkDotF32Generic(b *testing.B) {
	left := randomFloat32Slice(8192, 1)
	right := randomFloat32Slice(8192, 2)
	length := len(left)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		_ = DotF32Generic(&left[0], &right[0], length)
	}
}
