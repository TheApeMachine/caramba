package predictive_coding

import "unsafe"

func Prediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice((*float32)(weights), outDim*inDim)
	representationView := unsafe.Slice((*float32)(representation), inDim)
	outputView := unsafe.Slice((*float32)(output), outDim)

	for outIndex := 0; outIndex < outDim; outIndex++ {
		var sum float32

		for inIndex := 0; inIndex < inDim; inIndex++ {
			sum += weightsView[outIndex*inDim+inIndex] * representationView[inIndex]
		}

		outputView[outIndex] = sum
	}
}

func PredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	observedView := unsafe.Slice((*float32)(observed), count)
	predictedView := unsafe.Slice((*float32)(predicted), count)
	outputView := unsafe.Slice((*float32)(output), count)

	for index, value := range observedView {
		outputView[index] = value - predictedView[index]
	}
}

func UpdateRepresentation(
	config PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice((*float32)(weights), outDim*inDim)
	representationView := unsafe.Slice((*float32)(representation), inDim)
	errorView := unsafe.Slice((*float32)(predictionError), outDim)
	outputView := unsafe.Slice((*float32)(output), inDim)

	copy(outputView, representationView)

	for outIndex := 0; outIndex < outDim; outIndex++ {
		for inIndex := 0; inIndex < inDim; inIndex++ {
			outputView[inIndex] += config.LearningRate *
				weightsView[outIndex*inDim+inIndex] * errorView[outIndex]
		}
	}
}

func UpdateWeights(
	config PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice((*float32)(weights), outDim*inDim)
	representationView := unsafe.Slice((*float32)(representation), inDim)
	errorView := unsafe.Slice((*float32)(predictionError), outDim)
	outputView := unsafe.Slice((*float32)(output), outDim*inDim)

	copy(outputView, weightsView)

	for outIndex := 0; outIndex < outDim; outIndex++ {
		for inIndex := 0; inIndex < inDim; inIndex++ {
			outputView[outIndex*inDim+inIndex] +=
				config.LearningRate * errorView[outIndex] * representationView[inIndex]
		}
	}
}
