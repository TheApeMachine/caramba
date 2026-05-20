//go:build amd64

package active_inference

import "golang.org/x/sys/cpu"

func FreeEnergyFloat32Native(likelihood, posterior, prior []float32) float32 {
	if len(likelihood) == 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		return FreeEnergyF32AVX512(
			&likelihood[0], &posterior[0], &prior[0], len(likelihood),
		)
	}

	return FreeEnergyFloat32Scalar(likelihood, posterior, prior)
}

func ExpectedFreeEnergyFloat32Native(
	predictedObs, preferredObs, predictedState []float32,
) float32 {
	if len(predictedObs) == 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		return ExpectedFreeEnergyF32AVX512(
			&predictedObs[0], &preferredObs[0], &predictedState[0],
			len(predictedObs), len(predictedState),
		)
	}

	return ExpectedFreeEnergyFloat32Scalar(predictedObs, preferredObs, predictedState)
}

func BeliefUpdateFloat32Native(likelihood, prior, output []float32) {
	if len(likelihood) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		BeliefUpdateF32AVX512(&likelihood[0], &prior[0], &output[0], len(likelihood))
		return
	}

	BeliefUpdateFloat32Scalar(likelihood, prior, output)
}

func PrecisionWeightFloat32Native(errors, precision, output []float32) {
	if len(errors) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		PrecisionWeightF32AVX512(&errors[0], &precision[0], &output[0], len(errors))
		return
	}

	PrecisionWeightFloat32Scalar(errors, precision, output)
}
