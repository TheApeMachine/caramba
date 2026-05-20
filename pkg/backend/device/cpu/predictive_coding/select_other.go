//go:build !amd64

package predictive_coding

func PredictionFloat32Native(
	weights []float32,
	representation, output []float32,
	outDim, inDim int,
) {
	PredictionFloat32Scalar(weights, representation, output, outDim, inDim)
}

func PredictionErrorFloat32Native(observed, predicted, output []float32) {
	PredictionErrorFloat32Scalar(observed, predicted, output)
}

func UpdateRepresentationFloat32Native(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	UpdateRepresentationFloat32Scalar(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}

func UpdateWeightsFloat32Native(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	UpdateWeightsFloat32Scalar(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}
