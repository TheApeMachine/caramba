package active_inference

import "math"

const activeInferenceEps = 1e-12

/*
FreeEnergyFloat32Scalar computes F = E_q[-ln p(o|s)] + KL[q||p_prior].
*/
func FreeEnergyFloat32Scalar(likelihood, posterior, prior []float32) float32 {
	if len(likelihood) == 0 {
		return 0
	}

	var crossEntropy, kl float64

	for index, posteriorValue := range posterior {
		clampedLike := math.Max(activeInferenceEps, float64(likelihood[index]))
		clampedPosterior := math.Max(activeInferenceEps, float64(posteriorValue))
		clampedPrior := math.Max(activeInferenceEps, float64(prior[index]))

		crossEntropy += -float64(posteriorValue) * math.Log(clampedLike)
		kl += float64(posteriorValue) * (math.Log(clampedPosterior) - math.Log(clampedPrior))
	}

	return float32(crossEntropy + kl)
}

/*
ExpectedFreeEnergyFloat32Scalar computes G = pragmatic + epistemic terms.
*/
func ExpectedFreeEnergyFloat32Scalar(
	predictedObs, preferredObs, predictedState []float32,
) float32 {
	if len(predictedObs) == 0 {
		return 0
	}

	var pragmatic, epistemic float64

	for index, predicted := range predictedObs {
		predictedClamped := math.Max(activeInferenceEps, float64(predicted))
		preferredClamped := math.Max(activeInferenceEps, float64(preferredObs[index]))

		pragmatic += float64(predicted) * (math.Log(predictedClamped) - math.Log(preferredClamped))
	}

	for _, stateValue := range predictedState {
		clamped := math.Max(activeInferenceEps, float64(stateValue))
		epistemic += -float64(stateValue) * math.Log(clamped)
	}

	return float32(pragmatic + epistemic)
}

/*
pragmaticTermFloat32Scalar sums predicted * (log(predicted) - log(preferred)).
*/
func pragmaticTermFloat32Scalar(predictedObs, preferredObs []float32) float32 {
	var pragmatic float64

	for index, predicted := range predictedObs {
		predictedClamped := math.Max(activeInferenceEps, float64(predicted))
		preferredClamped := math.Max(activeInferenceEps, float64(preferredObs[index]))

		pragmatic += float64(predicted) * (math.Log(predictedClamped) - math.Log(preferredClamped))
	}

	return float32(pragmatic)
}

/*
epistemicTermFloat32Scalar sums -state * log(state).
*/
func epistemicTermFloat32Scalar(predictedState []float32) float32 {
	var epistemic float64

	for _, stateValue := range predictedState {
		clamped := math.Max(activeInferenceEps, float64(stateValue))
		epistemic += -float64(stateValue) * math.Log(clamped)
	}

	return float32(epistemic)
}

/*
BeliefUpdateFloat32Scalar writes posterior q(s|o) ∝ p(o|s) × q(s) and normalizes.
*/
func BeliefUpdateFloat32Scalar(likelihood, prior, output []float32) {
	if len(likelihood) == 0 {
		return
	}

	var sum float64

	for index, likeValue := range likelihood {
		product := likeValue * prior[index]
		output[index] = product
		sum += float64(product)
	}

	if sum == 0 {
		return
	}

	normalizer := float32(1.0 / sum)

	for index := range output {
		output[index] *= normalizer
	}
}

/*
PrecisionWeightFloat32Scalar writes output[i] = errors[i] * precision[i].
*/
func PrecisionWeightFloat32Scalar(errors, precision, output []float32) {
	for index, value := range errors {
		output[index] = value * precision[index]
	}
}
