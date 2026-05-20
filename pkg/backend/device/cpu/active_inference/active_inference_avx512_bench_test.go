//go:build amd64

package active_inference

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func requireAVX512Bench(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}
}

func BenchmarkFreeEnergyF32AVX512(b *testing.B) {
	requireAVX512Bench(b)

	length := 8192
	likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xC100)

	b.ResetTimer()

	for b.Loop() {
		_ = FreeEnergyF32AVX512(&likelihood[0], &posterior[0], &prior[0], length)
	}
}

func BenchmarkExpectedFreeEnergyF32AVX512(b *testing.B) {
	requireAVX512Bench(b)

	length := 8192
	predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(length, 0xC110)

	b.ResetTimer()

	for b.Loop() {
		_ = ExpectedFreeEnergyF32AVX512(
			&predictedObs[0], &preferredObs[0], &predictedState[0],
			length, length,
		)
	}
}

func BenchmarkBeliefUpdateF32AVX512(b *testing.B) {
	requireAVX512Bench(b)

	length := 8192
	likelihood, prior, _ := randomActiveInferenceVectors(length, 0xC120)
	output := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		BeliefUpdateF32AVX512(&likelihood[0], &prior[0], &output[0], length)
	}
}

func BenchmarkPrecisionWeightF32AVX512(b *testing.B) {
	requireAVX512Bench(b)

	length := 8192
	errors, precision, _ := randomActiveInferenceVectors(length, 0xC130)
	output := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		PrecisionWeightF32AVX512(&errors[0], &precision[0], &output[0], length)
	}
}
