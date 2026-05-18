//go:build arm64

package kernels

import "math"

// adamStepSlicesNEON is a thin wrapper around the NEON asm with a
// scalar tail for the remainder when len is not a multiple of 4. The
// original adamStepSlices in optimizers.go now delegates to this on
// arm64.
func adamStepSlicesNEON(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Corr := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Corr := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))

	n := len(params)
	blockN := n & ^3
	tailStart := blockN

	if blockN > 0 {
		adamStepFloat32NEONAsm(
			&params[0], &gradients[0], &firstMoment[0], &secondMoment[0], &output[0],
			blockN,
			config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
			beta1Corr, beta2Corr,
		)
	}

	// Scalar tail for the last <4 elements.
	for index := tailStart; index < n; index++ {
		gradValue := gradients[index]
		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue
		secondMoment[index] = config.Beta2*secondMoment[index] + (1-config.Beta2)*gradValue*gradValue
		biasFirst := firstMoment[index] / beta1Corr
		biasSec := secondMoment[index] / beta2Corr
		denom := float32(math.Sqrt(float64(biasSec))) + config.Epsilon
		output[index] = params[index] - config.LearningRate*biasFirst/denom
	}
}

func sgdStepSlicesNEON(
	config SGDConfig,
	params, gradients, momentum, output []float32,
) {
	n := len(params)
	blockN := n & ^3
	tailStart := blockN

	if blockN > 0 {
		sgdStepFloat32NEONAsm(
			&params[0], &gradients[0], &momentum[0], &output[0],
			blockN,
			config.LearningRate, config.Momentum, config.WeightDecay,
		)
	}

	for index := tailStart; index < n; index++ {
		effective := gradients[index] + config.WeightDecay*params[index]
		momentum[index] = config.Momentum*momentum[index] + effective
		update := momentum[index]
		if config.Nesterov {
			update = effective + config.Momentum*momentum[index]
		}
		output[index] = params[index] - config.LearningRate*update
	}
}
