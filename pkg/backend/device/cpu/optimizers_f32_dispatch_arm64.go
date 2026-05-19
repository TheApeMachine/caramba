//go:build arm64

package cpu

import "math"

// adamStepSlices on arm64 dispatches to the NEON inner loop.
func adamStepSlices(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	adamStepSlicesNEON(config, params, gradients, firstMoment, secondMoment, output)
}

func adamWStepSlices(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	adamwStepSlicesNEON(config, params, gradients, firstMoment, secondMoment, output)
}

func sgdStepSlices(config SGDConfig, params, gradients, momentum, output []float32) {
	sgdStepSlicesNEON(config, params, gradients, momentum, output)
}

func adamaxStepSlices(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	adamaxStepSlicesNEON(config, params, gradients, firstMoment, infinityMoment, output)
}

func adagradStepSlices(config AdagradConfig, params, gradients, accumulator, output []float32) {
	adagradStepSlicesNEON(config, params, gradients, accumulator, output)
}

func rmspropStepSlices(config RMSpropConfig, params, gradients, secondMoment, output []float32) {
	rmspropStepSlicesNEON(config, params, gradients, secondMoment, output)
}

func lionStepSlices(config LionConfig, params, gradients, momentum, output []float32) {
	lionStepSlicesNEON(config, params, gradients, momentum, output)
}

func larsStepSlices(config LARSConfig, params, gradients, momentum, output []float32) {
	larsStepSlicesNEON(config, params, gradients, momentum, output)
}

func lbfgsStepSlices(config LBFGSConfig, params, gradients, output []float32) {
	lbfgsStepSlicesNEON(config, params, gradients, output)
}

func hebbianStepSlices(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	hebbianStepSlicesNEON(config, weights, post, pre, output, preDim)
}

func hebbianStepSlicesNEON(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	decayFactor := float32(1 - config.Decay)

	for postIndex, postValue := range post {
		rowStart := postIndex * preDim
		weightsRow := weights[rowStart : rowStart+preDim]
		outRow := output[rowStart : rowStart+preDim]
		lrPost := config.LearningRate * postValue
		blockCount := preDim &^ 3

		if blockCount > 0 {
			HebbianStepRowFloat32NEONAsm(
				&weightsRow[0], &pre[0], &outRow[0],
				blockCount, decayFactor, lrPost,
			)
		}

		for index := blockCount; index < preDim; index++ {
			outRow[index] = weightsRow[index]*decayFactor + lrPost*pre[index]
		}
	}
}

// AdamStepSlicesNEON is a thin wrapper around the NEON asm with a
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
		AdamStepFloat32NEONAsm(
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
		AdamwStepFloat32NEONAsm(
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
		AdamaxStepFloat32NEONAsm(
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
		AdagradStepFloat32NEONAsm(
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
		RmspropStepFloat32NEONAsm(
			&params[0], &gradients[0], &secondMoment[0], &output[0],
			blockN, config.LearningRate, config.Decay, config.Epsilon,
		)
	}

	for index := tailStart; index < n; index++ {
		gradValue := gradients[index]
		gradSquared := gradValue * gradValue
		scaledGrad := (1 - config.Decay) * gradSquared
		secondMoment[index] = scaledGrad + config.Decay*secondMoment[index]
		denom := optimizerSqrtFloat32(secondMoment[index]) + config.Epsilon
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
		SgdStepFloat32NEONAsm(
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

func lionStepSlicesNEON(
	config LionConfig,
	params, gradients, momentum, output []float32,
) {
	n := len(params)
	blockN := n & ^3

	if blockN > 0 {
		LionStepFloat32NEONAsm(
			&params[0], &gradients[0], &momentum[0], &output[0],
			blockN, config.LearningRate, config.Beta1, config.Beta2, config.WeightDecay,
		)
	}

	for index := blockN; index < n; index++ {
		gradValue := gradients[index]
		update := config.Beta1*momentum[index] + (1-config.Beta1)*gradValue
		sign := float32(0)

		if update > 0 {
			sign = 1
		}

		if update < 0 {
			sign = -1
		}

		decayStep := config.WeightDecay * params[index]
		output[index] = params[index] - config.LearningRate*(sign+decayStep)
		momentum[index] = config.Beta2*momentum[index] + (1-config.Beta2)*gradValue
	}
}

func lbfgsStepSlicesNEON(config LBFGSConfig, params, gradients, output []float32) {
	n := len(params)
	blockN := n & ^3

	if blockN > 0 {
		LbfgsStepFloat32NEONAsm(&params[0], &gradients[0], &output[0], blockN, config.LearningRate)
	}

	for index := blockN; index < n; index++ {
		output[index] = params[index] - config.LearningRate*gradients[index]
	}
}

func larsStepSlicesNEON(
	config LARSConfig,
	params, gradients, momentum, output []float32,
) {
	effectiveLr := larsEffectiveLearningRate(config, params, gradients)
	n := len(params)
	blockN := n & ^3

	if blockN > 0 {
		LarsStepFloat32NEONAsm(
			&params[0], &gradients[0], &momentum[0], &output[0],
			blockN, config.LearningRate, config.Momentum, config.WeightDecay, effectiveLr,
		)
	}

	for index := blockN; index < n; index++ {
		effective := gradients[index] + config.WeightDecay*params[index]
		momentum[index] = config.Momentum*momentum[index] + effective
		output[index] = params[index] - effectiveLr*momentum[index]
	}
}

func larsEffectiveLearningRate(config LARSConfig, params []float32, gradients []float32) float32 {
	var paramsNorm, gradsNorm float64

	for index, value := range params {
		paramsNorm += float64(value) * float64(value)
		gradsNorm += float64(gradients[index]) * float64(gradients[index])
	}

	paramsNorm = math.Sqrt(paramsNorm)
	gradsNorm = math.Sqrt(gradsNorm)

	if paramsNorm == 0 || gradsNorm == 0 {
		return config.LearningRate
	}

	trust := config.TrustCoeff *
		float32(paramsNorm) /
		(float32(gradsNorm) + config.WeightDecay*float32(paramsNorm) + config.Epsilon)
	return config.LearningRate * trust
}
