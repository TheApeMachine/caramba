//go:build amd64

package rope

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkRopePairsF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const pairCount = 4096

	in, cos, sin := randomRopePairBuffers(pairCount, 0x5210)
	out := make([]float32, 2*pairCount)

	b.SetBytes(int64(2 * pairCount * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		ropePairsF32AVX512(out, in, cos, sin)
	}
}

func BenchmarkRopePairsF32Generic(b *testing.B) {
	const pairCount = 4096

	in, cos, sin := randomRopePairBuffers(pairCount, 0x5211)
	out := make([]float32, 2*pairCount)

	b.SetBytes(int64(2 * pairCount * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		RopePairsGeneric(out, in, cos, sin)
	}
}
