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

func adamwStepSlicesNEON(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Corr := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Corr := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))

	n := len(params)
	blockN := n & ^3
	tailStart := blockN

	if blockN > 0 {
		adamwStepFloat32NEONAsm(
			&params[0], &gradients[0], &firstMoment[0], &secondMoment[0], &output[0],
			blockN,
			config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
			beta1Corr, beta2Corr, config.WeightDecay,
		)
	}

	for index := tailStart; index < n; index++ {
		gradValue := gradients[index]
		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue
		secondMoment[index] = config.Beta2*secondMoment[index] + (1-config.Beta2)*gradValue*gradValue
		biasFirst := firstMoment[index] / beta1Corr
		biasSec := secondMoment[index] / beta2Corr
		denom := float32(math.Sqrt(float64(biasSec))) + config.Epsilon
		gradStep := config.LearningRate * biasFirst / denom
		decayStep := config.LearningRate * config.WeightDecay * params[index]
		output[index] = params[index] - gradStep - decayStep
	}
}

func adamaxStepSlicesNEON(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	beta1Corr := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	n := len(params)
	blockN := n & ^3
	tailStart := blockN

	if blockN > 0 {
		adamaxStepFloat32NEONAsm(
			&params[0], &gradients[0], &firstMoment[0], &infinityMoment[0], &output[0],
			blockN,
			config.LearningRate, config.Beta1, config.Beta2, config.Epsilon, beta1Corr,
		)
	}

	for index := tailStart; index < n; index++ {
		gradValue := gradients[index]
		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue
		updated := config.Beta2 * infinityMoment[index]
		absGrad := float32(math.Abs(float64(gradValue)))
		if absGrad > updated {
			updated = absGrad
		}
		infinityMoment[index] = updated
		biasFirst := firstMoment[index] / beta1Corr
		output[index] = params[index] - config.LearningRate*biasFirst/(infinityMoment[index]+config.Epsilon)
	}
}

func adagradStepSlicesNEON(
	config AdagradConfig,
	params, gradients, accumulator, output []float32,
) {
	n := len(params)
	blockN := n & ^3
	tailStart := blockN

	if blockN > 0 {
		adagradStepFloat32NEONAsm(
			&params[0], &gradients[0], &accumulator[0], &output[0],
			blockN, config.LearningRate, config.Epsilon,
		)
	}

	for index := tailStart; index < n; index++ {
		gradValue := gradients[index]
		accumulator[index] += gradValue * gradValue
		denom := float32(math.Sqrt(float64(accumulator[index]))) + config.Epsilon
		output[index] = params[index] - config.LearningRate*gradValue/denom
	}
}

func rmspropStepSlicesNEON(
	config RMSpropConfig,
	params, gradients, secondMoment, output []float32,
) {
	n := len(params)
	blockN := n & ^3
	tailStart := blockN

	if blockN > 0 {
		rmspropStepFloat32NEONAsm(
			&params[0], &gradients[0], &secondMoment[0], &output[0],
			blockN, config.LearningRate, config.Decay, config.Epsilon,
		)
	}

	for index := tailStart; index < n; index++ {
		gradValue := gradients[index]
		secondMoment[index] = config.Decay*secondMoment[index] + (1-config.Decay)*gradValue*gradValue
		denom := float32(math.Sqrt(float64(secondMoment[index]))) + config.Epsilon
		output[index] = params[index] - config.LearningRate*gradValue/denom
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
