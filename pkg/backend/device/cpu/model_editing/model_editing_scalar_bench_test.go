package model_editing

import "testing"

func BenchmarkWeightGraftAddFloat32Scalar(b *testing.B) {
	const benchLen = 8192

	weights, injection := randomGraftVectors(benchLen, 0x2B30)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		WeightGraftAddFloat32Scalar(weights, injection)
	}
}
