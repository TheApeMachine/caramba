//go:build amd64

package model_editing

import "testing"

func BenchmarkWeightGraftAddFloat32AVX512(b *testing.B) {
	const benchLen = 8192

	weights, injection := randomGraftVectors(benchLen, 0x2B31)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		WeightGraftAddFloat32AVX512(&weights[0], &injection[0], benchLen)
	}
}
