//go:build amd64

package sampling

import (
	"testing"
)

func BenchmarkGreedySampleF32AVX512(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x1620)

	b.ResetTimer()

	for b.Loop() {
		GreedySampleF32AVX512(&logits[0], len(logits))
	}
}

func BenchmarkSamplingSoftmaxRowF32AVX512(b *testing.B) {
	length := 8192
	logits := randomSamplingLogits(length, 0x1621)
	out := make([]float32, length)
	temperature := float32(1.0)

	b.ResetTimer()

	for b.Loop() {
		SamplingSoftmaxRowF32AVX512(&logits[0], &out[0], temperature, length)
	}
}
