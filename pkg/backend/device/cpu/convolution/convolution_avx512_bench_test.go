//go:build amd64

package convolution

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkConvPatchDotF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	weight, patch := randomConvolutionFloat32Pair(8192, 0xC0C)
	length := len(weight)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		_ = ConvPatchDotF32AVX512(&weight[0], &patch[0], length)
	}
}

func BenchmarkConvPatchDotScalar(b *testing.B) {
	weight, patch := randomConvolutionFloat32Pair(8192, 0xC0D)
	length := len(weight)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		_ = ConvPatchDotScalar(weight, patch, length)
	}
}
