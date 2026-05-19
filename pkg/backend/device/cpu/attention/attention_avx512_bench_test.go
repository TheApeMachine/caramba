//go:build amd64

package attention

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkFlashAttentionOnlineUpdateAVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	acc := randomAttentionFloat32Slice(8192, 0xA76)
	valueRow := randomAttentionFloat32Slice(8192, 0xA77)
	length := len(acc)
	alpha := float32(0.5)
	shifted := float32(0.25)

	b.SetBytes(int64(length * 4 * 3))
	b.ResetTimer()

	for b.Loop() {
		flashAttentionOnlineUpdateAVX512(&acc[0], &valueRow[0], alpha, shifted, length)
	}
}

func BenchmarkFlashAttentionOnlineUpdateScalar(b *testing.B) {
	acc := randomAttentionFloat32Slice(8192, 0xA78)
	valueRow := randomAttentionFloat32Slice(8192, 0xA79)
	length := len(acc)
	alpha := float32(0.5)
	shifted := float32(0.25)

	b.SetBytes(int64(length * 4 * 3))
	b.ResetTimer()

	for b.Loop() {
		flashOnlineUpdateScalar(acc, valueRow, alpha, shifted, length)
	}
}

func BenchmarkFlashAttentionScaleAVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	acc := randomAttentionFloat32Slice(8192, 0xA7A)
	out := make([]float32, 8192)
	length := len(acc)
	invNormalizer := float32(0.125)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		flashAttentionScaleAVX512(&out[0], &acc[0], invNormalizer, length)
	}
}
