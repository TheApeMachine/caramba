package predictive_coding

/*
PredictionFloat32Scalar computes p̂ = W × s for row-major weights [outDim, inDim].
*/
func PredictionFloat32Scalar(
	weights []float32,
	representation, output []float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	for outIndex := 0; outIndex < outDim; outIndex++ {
		rowOffset := outIndex * inDim
		var sum float64

		for inIndex := 0; inIndex < inDim; inIndex++ {
			sum += float64(weights[rowOffset+inIndex]) *
				float64(representation[inIndex])
		}

		output[outIndex] = float32(sum)
	}
}

/*
PredictionErrorFloat32Scalar computes e = observed − predicted element-wise.
*/
func PredictionErrorFloat32Scalar(observed, predicted, output []float32) {
	for index, value := range observed {
		output[index] = value - predicted[index]
	}
}

/*
UpdateRepresentationFloat32Scalar computes s_new = s + lr × W^T × e.
*/
func UpdateRepresentationFloat32Scalar(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	copy(output, representation)

	for outIndex := 0; outIndex < outDim; outIndex++ {
		rowOffset := outIndex * inDim
		scale := float64(learningRate) * float64(predictionError[outIndex])

		for inIndex := 0; inIndex < inDim; inIndex++ {
			output[inIndex] += float32(
				scale * float64(weights[rowOffset+inIndex]),
			)
		}
	}
}

/*
UpdateWeightsFloat32Scalar computes W_new = W + lr × outer(e, s).
*/
func UpdateWeightsFloat32Scalar(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	copy(output, weights)

	for outIndex := 0; outIndex < outDim; outIndex++ {
		rowOffset := outIndex * inDim
		scale := float64(learningRate) * float64(predictionError[outIndex])

		for inIndex := 0; inIndex < inDim; inIndex++ {
			output[rowOffset+inIndex] += float32(
				scale * float64(representation[inIndex]),
			)
		}
	}
}
