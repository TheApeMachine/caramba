package sampling

import "testing"

func BenchmarkGreedySampleFloat32Native(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x1630)

	b.ResetTimer()

	for b.Loop() {
		GreedySampleFloat32Native(logits)
	}
}

func BenchmarkSamplingSoftmaxRowFloat32Native(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x1631)
	out := make([]float32, len(logits))
	temperature := float32(1.0)

	b.ResetTimer()

	for b.Loop() {
		SamplingSoftmaxRowFloat32Native(logits, out, temperature)
	}
}
