//go:build amd64

package predictive_coding

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkPCPredictionF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchDim = 1024
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xBC11)
	_, representation, _ := randomPredictiveCodingVectors(benchDim, 0xBC12)
	output := make([]float32, benchDim)

	for b.Loop() {
		PCPredictionF32AVX512(
			&weights[0], &representation[0], &output[0], benchDim, benchDim,
		)
	}
}

func BenchmarkPCPredictionErrorF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	observed, predicted, _ := randomPredictiveCodingVectors(benchLen, 0xBC13)
	output := make([]float32, benchLen)

	for b.Loop() {
		PCPredictionErrorF32AVX512(
			&observed[0], &predicted[0], &output[0], benchLen,
		)
	}
}

func BenchmarkPCUpdateRepresentationF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchDim = 64
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xBC14)
	_, representation, predictionError := randomPredictiveCodingVectors(benchDim, 0xBC15)
	output := make([]float32, benchDim)

	for b.Loop() {
		PCUpdateRepresentationF32AVX512(
			&weights[0], &representation[0], &predictionError[0], &output[0],
			predictiveCodingTestLR, benchDim, benchDim,
		)
	}
}

func BenchmarkPCUpdateWeightsF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchDim = 64
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xBC16)
	_, representation, predictionError := randomPredictiveCodingVectors(benchDim, 0xBC17)
	output := make([]float32, benchDim*benchDim)

	for b.Loop() {
		PCUpdateWeightsF32AVX512(
			&weights[0], &representation[0], &predictionError[0], &output[0],
			predictiveCodingTestLR, benchDim, benchDim,
		)
	}
}
