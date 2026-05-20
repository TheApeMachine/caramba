package predictive_coding

import "unsafe"

func PredictionF32Generic(
	weights, representation, output *float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice(weights, outDim*inDim)
	representationView := unsafe.Slice(representation, inDim)
	outputView := unsafe.Slice(output, outDim)

	PredictionFloat32Scalar(weightsView, representationView, outputView, outDim, inDim)
}

func PredictionErrorF32Generic(observed, predicted, output *float32, count int) {
	if count == 0 {
		return
	}

	observedView := unsafe.Slice(observed, count)
	predictedView := unsafe.Slice(predicted, count)
	outputView := unsafe.Slice(output, count)

	PredictionErrorFloat32Scalar(observedView, predictedView, outputView)
}

func UpdateRepresentationF32Generic(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice(weights, outDim*inDim)
	representationView := unsafe.Slice(representation, inDim)
	errorView := unsafe.Slice(predictionError, outDim)
	outputView := unsafe.Slice(output, inDim)

	UpdateRepresentationFloat32Scalar(
		weightsView, representationView, errorView, outputView,
		learningRate, outDim, inDim,
	)
}

func UpdateWeightsF32Generic(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice(weights, outDim*inDim)
	representationView := unsafe.Slice(representation, inDim)
	errorView := unsafe.Slice(predictionError, outDim)
	outputView := unsafe.Slice(output, outDim*inDim)

	UpdateWeightsFloat32Scalar(
		weightsView, representationView, errorView, outputView,
		learningRate, outDim, inDim,
	)
}
