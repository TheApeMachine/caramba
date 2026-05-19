package active_inference

import (
	"math"
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

	const eps = 1e-12

	var crossEntropy, kl float64

	for index, posteriorValue := range posteriorView {
		clampedLike := math.Max(eps, float64(likelihoodView[index]))
		clampedPosterior := math.Max(eps, float64(posteriorValue))
		clampedPrior := math.Max(eps, float64(priorView[index]))

		crossEntropy += -float64(posteriorValue) * math.Log(clampedLike)
		kl += float64(posteriorValue) * (math.Log(clampedPosterior) - math.Log(clampedPrior))
	}

	outputView[0] = float32(crossEntropy + kl)
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

	const eps = 1e-12

	var pragmatic, epistemic float64

	for index, predicted := range predictedObsView {
		predictedClamped := math.Max(eps, float64(predicted))
		preferredClamped := math.Max(eps, float64(preferredObsView[index]))

		pragmatic += float64(predicted) * (math.Log(predictedClamped) - math.Log(preferredClamped))
	}

	for _, stateValue := range predictedStateView {
		clamped := math.Max(eps, float64(stateValue))
		epistemic += -float64(stateValue) * math.Log(clamped)
	}

	outputView[0] = float32(pragmatic + epistemic)
}

func BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	requireActiveInferenceFloat32(format)

	if count == 0 {
		return
	}

	likelihoodView := unsafe.Slice((*float32)(likelihood), count)
	priorView := unsafe.Slice((*float32)(prior), count)
	outputView := unsafe.Slice((*float32)(output), count)

	var sum float64

	for index, likeValue := range likelihoodView {
		product := likeValue * priorView[index]
		outputView[index] = product
		sum += float64(product)
	}

	if sum == 0 {
		return
	}

	normalizer := 1.0 / float32(sum)

	for index := range outputView {
		outputView[index] *= normalizer
	}
}

func PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	requireActiveInferenceFloat32(format)

	if count == 0 {
		return
	}

	errorsView := unsafe.Slice((*float32)(errors), count)
	precisionView := unsafe.Slice((*float32)(precision), count)
	outputView := unsafe.Slice((*float32)(output), count)

	for index, value := range errorsView {
		outputView[index] = value * precisionView[index]
	}
}
