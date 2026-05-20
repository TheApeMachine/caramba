//go:build !amd64

package active_inference

func FreeEnergyFloat32Native(likelihood, posterior, prior []float32) float32 {
	return FreeEnergyFloat32Scalar(likelihood, posterior, prior)
}

func ExpectedFreeEnergyFloat32Native(
	predictedObs, preferredObs, predictedState []float32,
) float32 {
	return ExpectedFreeEnergyFloat32Scalar(predictedObs, preferredObs, predictedState)
}

func BeliefUpdateFloat32Native(likelihood, prior, output []float32) {
	BeliefUpdateFloat32Scalar(likelihood, prior, output)
}

func PrecisionWeightFloat32Native(errors, precision, output []float32) {
	PrecisionWeightFloat32Scalar(errors, precision, output)
}
