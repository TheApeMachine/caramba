package active_inference

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func requireActiveInferenceFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("active_inference: unsupported dtype")
	}
}

func FreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireActiveInferenceFloat32(format)

	if count == 0 {
		return
	}

	likelihoodView := unsafe.Slice((*float32)(likelihood), count)
	posteriorView := unsafe.Slice((*float32)(posterior), count)
	priorView := unsafe.Slice((*float32)(prior), count)
	outputView := unsafe.Slice((*float32)(output), 1)

	outputView[0] = FreeEnergyFloat32Native(likelihoodView, posteriorView, priorView)
}

func ExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
) {
	requireActiveInferenceFloat32(format)

	if obsCount == 0 {
		return
	}

	predictedObsView := unsafe.Slice((*float32)(predictedObs), obsCount)
	preferredObsView := unsafe.Slice((*float32)(preferredObs), obsCount)
	predictedStateView := unsafe.Slice((*float32)(predictedState), stateCount)
	outputView := unsafe.Slice((*float32)(output), 1)

	outputView[0] = ExpectedFreeEnergyFloat32Native(
		predictedObsView, preferredObsView, predictedStateView,
	)
}

func BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	requireActiveInferenceFloat32(format)

	if count == 0 {
		return
	}

	likelihoodView := unsafe.Slice((*float32)(likelihood), count)
	priorView := unsafe.Slice((*float32)(prior), count)
	outputView := unsafe.Slice((*float32)(output), count)

	BeliefUpdateFloat32Native(likelihoodView, priorView, outputView)
}

func PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	requireActiveInferenceFloat32(format)

	if count == 0 {
		return
	}

	errorsView := unsafe.Slice((*float32)(errors), count)
	precisionView := unsafe.Slice((*float32)(precision), count)
	outputView := unsafe.Slice((*float32)(output), count)

	PrecisionWeightFloat32Native(errorsView, precisionView, outputView)
}
