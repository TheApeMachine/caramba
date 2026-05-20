package predictive_coding

import (
	"testing"
)

func BenchmarkPredictionFloat32Scalar(b *testing.B) {
	const benchDim = 1024
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xBC01)
	_, representation, _ := randomPredictiveCodingVectors(benchDim, 0xBC02)
	output := make([]float32, benchDim)

	for b.Loop() {
		PredictionFloat32Scalar(weights, representation, output, benchDim, benchDim)
	}
}

func BenchmarkPredictionErrorFloat32Scalar(b *testing.B) {
	const benchLen = 8192
	observed, predicted, _ := randomPredictiveCodingVectors(benchLen, 0xBC03)
	output := make([]float32, benchLen)

	for b.Loop() {
		PredictionErrorFloat32Scalar(observed, predicted, output)
	}
}

func BenchmarkUpdateRepresentationFloat32Scalar(b *testing.B) {
	const benchDim = 64
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xBC04)
	_, representation, predictionError := randomPredictiveCodingVectors(benchDim, 0xBC05)
	output := make([]float32, benchDim)

	for b.Loop() {
		UpdateRepresentationFloat32Scalar(
			weights, representation, predictionError, output,
			predictiveCodingTestLR, benchDim, benchDim,
		)
	}
}

func BenchmarkUpdateWeightsFloat32Scalar(b *testing.B) {
	const benchDim = 64
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xBC06)
	_, representation, predictionError := randomPredictiveCodingVectors(benchDim, 0xBC07)
	output := make([]float32, benchDim*benchDim)

	for b.Loop() {
		UpdateWeightsFloat32Scalar(
			weights, representation, predictionError, output,
			predictiveCodingTestLR, benchDim, benchDim,
		)
	}
}
