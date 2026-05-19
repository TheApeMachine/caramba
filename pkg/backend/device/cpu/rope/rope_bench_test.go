package rope

import "testing"

func BenchmarkRopePairsF32GenericHost(b *testing.B) {
	const pairCount = 4096

	in, cos, sin := randomRopePairBuffers(pairCount, 0x5212)
	out := make([]float32, 2*pairCount)

	b.SetBytes(int64(2 * pairCount * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		RopePairsGeneric(out, in, cos, sin)
	}
}
