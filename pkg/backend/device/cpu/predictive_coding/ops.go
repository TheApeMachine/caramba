package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func requirePredictiveCodingFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("predictive_coding: unsupported dtype")
	}
}

func Prediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	requirePredictiveCodingFloat32(format)

	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice((*float32)(weights), outDim*inDim)
	representationView := unsafe.Slice((*float32)(representation), inDim)
	outputView := unsafe.Slice((*float32)(output), outDim)

	PredictionFloat32Native(weightsView, representationView, outputView, outDim, inDim)
}

func PredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requirePredictiveCodingFloat32(format)

	if count == 0 {
		return
	}

	observedView := unsafe.Slice((*float32)(observed), count)
	predictedView := unsafe.Slice((*float32)(predicted), count)
	outputView := unsafe.Slice((*float32)(output), count)

	PredictionErrorFloat32Native(observedView, predictedView, outputView)
}

func UpdateRepresentation(
	config PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	requirePredictiveCodingFloat32(format)

	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice((*float32)(weights), outDim*inDim)
	representationView := unsafe.Slice((*float32)(representation), inDim)
	errorView := unsafe.Slice((*float32)(predictionError), outDim)
	outputView := unsafe.Slice((*float32)(output), inDim)

	UpdateRepresentationFloat32Native(
		weightsView, representationView, errorView, outputView,
		config.LearningRate, outDim, inDim,
	)
}

func UpdateWeights(
	config PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	requirePredictiveCodingFloat32(format)

	if outDim == 0 || inDim == 0 {
		return
	}

	weightsView := unsafe.Slice((*float32)(weights), outDim*inDim)
	representationView := unsafe.Slice((*float32)(representation), inDim)
	errorView := unsafe.Slice((*float32)(predictionError), outDim)
	outputView := unsafe.Slice((*float32)(output), outDim*inDim)

	UpdateWeightsFloat32Native(
		weightsView, representationView, errorView, outputView,
		config.LearningRate, outDim, inDim,
	)
}
