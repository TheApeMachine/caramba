package active_inference

import "unsafe"

func FreeEnergyF32Generic(likelihood, posterior, prior *float32, count int) float32 {
	likelihoodView := unsafe.Slice(likelihood, count)
	posteriorView := unsafe.Slice(posterior, count)
	priorView := unsafe.Slice(prior, count)

	return FreeEnergyFloat32Scalar(likelihoodView, posteriorView, priorView)
}

func ExpectedFreeEnergyF32Generic(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32 {
	predictedObsView := unsafe.Slice(predictedObs, obsCount)
	preferredObsView := unsafe.Slice(preferredObs, obsCount)
	predictedStateView := unsafe.Slice(predictedState, stateCount)

	return ExpectedFreeEnergyFloat32Scalar(
		predictedObsView, preferredObsView, predictedStateView,
	)
}

func BeliefUpdateF32Generic(
	likelihood, prior, output *float32,
	count int,
) {
	likelihoodView := unsafe.Slice(likelihood, count)
	priorView := unsafe.Slice(prior, count)
	outputView := unsafe.Slice(output, count)

	BeliefUpdateFloat32Scalar(likelihoodView, priorView, outputView)
}

func PrecisionWeightF32Generic(errors, precision, output *float32, count int) {
	errorsView := unsafe.Slice(errors, count)
	precisionView := unsafe.Slice(precision, count)
	outputView := unsafe.Slice(output, count)

	PrecisionWeightFloat32Scalar(errorsView, precisionView, outputView)
}
