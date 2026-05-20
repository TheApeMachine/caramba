package active_inference

import (
	"testing"
)

func BenchmarkFreeEnergyFloat32Scalar(b *testing.B) {
	length := 8192
	likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xB100)

	b.ResetTimer()

	for b.Loop() {
		_ = FreeEnergyFloat32Scalar(likelihood, posterior, prior)
	}
}

func BenchmarkExpectedFreeEnergyFloat32Scalar(b *testing.B) {
	length := 8192
	predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(length, 0xB110)

	b.ResetTimer()

	for b.Loop() {
		_ = ExpectedFreeEnergyFloat32Scalar(predictedObs, preferredObs, predictedState)
	}
}

func BenchmarkBeliefUpdateFloat32Scalar(b *testing.B) {
	length := 8192
	likelihood, prior, _ := randomActiveInferenceVectors(length, 0xB120)
	output := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		BeliefUpdateFloat32Scalar(likelihood, prior, output)
	}
}

func BenchmarkPrecisionWeightFloat32Scalar(b *testing.B) {
	length := 8192
	errors, precision, _ := randomActiveInferenceVectors(length, 0xB130)
	output := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		PrecisionWeightFloat32Scalar(errors, precision, output)
	}
}
