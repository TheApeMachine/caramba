//go:build amd64

package optimizer

import (
	"math"

	"golang.org/x/sys/cpu"
)

func adamStepSlices(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	if cpu.X86.HasAVX512F {
		adamStepSlicesAVX512(config, params, gradients, firstMoment, secondMoment, output)
		return
	}

	adamStepSlicesScalar(config, params, gradients, firstMoment, secondMoment, output)
}

func adamWStepSlices(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	if cpu.X86.HasAVX512F {
		adamWStepSlicesAVX512(config, params, gradients, firstMoment, secondMoment, output)
		return
	}

	adamWStepSlicesScalar(config, params, gradients, firstMoment, secondMoment, output)
}

func sgdStepSlices(config SGDConfig, params, gradients, momentum, output []float32) {
	if config.Nesterov || !cpu.X86.HasAVX512F {
		sgdStepSlicesScalar(config, params, gradients, momentum, output)
		return
	}

	sgdStepSlicesAVX512(config, params, gradients, momentum, output)
}

func adamaxStepSlices(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	if cpu.X86.HasAVX512F {
		adamaxStepSlicesAVX512(config, params, gradients, firstMoment, infinityMoment, output)
		return
	}

	adamaxStepSlicesScalar(config, params, gradients, firstMoment, infinityMoment, output)
}

func adagradStepSlices(config AdagradConfig, params, gradients, accumulator, output []float32) {
	if cpu.X86.HasAVX512F {
		adagradStepSlicesAVX512(config, params, gradients, accumulator, output)
		return
	}

	adagradStepSlicesScalar(config, params, gradients, accumulator, output)
}

func rmspropStepSlices(config RMSpropConfig, params, gradients, secondMoment, output []float32) {
	if cpu.X86.HasAVX512F {
		rmspropStepSlicesAVX512(config, params, gradients, secondMoment, output)
		return
	}

	rmspropStepSlicesScalar(config, params, gradients, secondMoment, output)
}

func lionStepSlices(config LionConfig, params, gradients, momentum, output []float32) {
	if cpu.X86.HasAVX512F {
		lionStepSlicesAVX512(config, params, gradients, momentum, output)
		return
	}

	lionStepSlicesScalar(config, params, gradients, momentum, output)
}

func larsStepSlices(config LARSConfig, params, gradients, momentum, output []float32) {
	if cpu.X86.HasAVX512F {
		larsStepSlicesAVX512(config, params, gradients, momentum, output)
		return
	}

	larsStepSlicesScalar(config, params, gradients, momentum, output)
}

func lbfgsStepSlices(config LBFGSConfig, params, gradients, output []float32) {
	if cpu.X86.HasAVX512F {
		lbfgsStepSlicesAVX512(config, params, gradients, output)
		return
	}

	lbfgsStepSlicesScalar(config, params, gradients, output)
}

func hebbianStepSlices(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	if cpu.X86.HasAVX512F {
		hebbianStepSlicesAVX512(config, weights, post, pre, output, preDim)
		return
	}

	hebbianStepSlicesScalar(config, weights, post, pre, output, preDim)
}

func adamStepSlicesAVX512(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))
	count := len(params)

	if count == 0 {
		return
	}

	AdamStepFloat32AVX512Asm(
		&params[0], &gradients[0], &firstMoment[0], &secondMoment[0], &output[0],
		count,
		config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
		beta1Correction, beta2Correction,
	)
}

func adamWStepSlicesAVX512(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))
	count := len(params)

	if count == 0 {
		return
	}

	AdamwStepFloat32AVX512Asm(
		&params[0], &gradients[0], &firstMoment[0], &secondMoment[0], &output[0],
		count,
		config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
		beta1Correction, beta2Correction, config.WeightDecay,
	)
}

func sgdStepSlicesAVX512(
	config SGDConfig,
	params, gradients, momentum, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	SgdStepFloat32AVX512Asm(
		&params[0], &gradients[0], &momentum[0], &output[0],
		count,
		config.LearningRate, config.Momentum, config.WeightDecay,
	)
}

func adamaxStepSlicesAVX512(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	count := len(params)

	if count == 0 {
		return
	}

	AdamaxStepFloat32AVX512Asm(
		&params[0], &gradients[0], &firstMoment[0], &infinityMoment[0], &output[0],
		count,
		config.LearningRate, config.Beta1, config.Beta2, config.Epsilon, beta1Correction,
	)
}

func adagradStepSlicesAVX512(
	config AdagradConfig,
	params, gradients, accumulator, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	AdagradStepFloat32AVX512Asm(
		&params[0], &gradients[0], &accumulator[0], &output[0],
		count, config.LearningRate, config.Epsilon,
	)
}

func rmspropStepSlicesAVX512(
	config RMSpropConfig,
	params, gradients, secondMoment, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	RmspropStepFloat32AVX512Asm(
		&params[0], &gradients[0], &secondMoment[0], &output[0],
		count, config.LearningRate, config.Decay, config.Epsilon,
	)
}

func lionStepSlicesAVX512(
	config LionConfig,
	params, gradients, momentum, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	LionStepFloat32AVX512Asm(
		&params[0], &gradients[0], &momentum[0], &output[0],
		count, config.LearningRate, config.Beta1, config.Beta2, config.WeightDecay,
	)
}

func lbfgsStepSlicesAVX512(config LBFGSConfig, params, gradients, output []float32) {
	count := len(params)

	if count == 0 {
		return
	}

	LbfgsStepFloat32AVX512Asm(
		&params[0], &gradients[0], &output[0],
		count, config.LearningRate,
	)
}

func larsStepSlicesAVX512(
	config LARSConfig,
	params, gradients, momentum, output []float32,
) {
	effectiveLearningRate := larsEffectiveLearningRate(config, params, gradients)
	count := len(params)

	if count == 0 {
		return
	}

	LarsStepFloat32AVX512Asm(
		&params[0], &gradients[0], &momentum[0], &output[0],
		count, config.LearningRate, config.Momentum, config.WeightDecay, effectiveLearningRate,
	)
}

func hebbianStepSlicesAVX512(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	decayFactor := float32(1 - config.Decay)

	for postIndex, postValue := range post {
		rowStart := postIndex * preDim
		weightsRow := weights[rowStart : rowStart+preDim]
		outRow := output[rowStart : rowStart+preDim]
		learningRatePost := config.LearningRate * postValue
		rowCount := len(weightsRow)

		if rowCount == 0 {
			continue
		}

		HebbianStepRowFloat32AVX512Asm(
			&weightsRow[0], &pre[0], &outRow[0],
			rowCount, decayFactor, learningRatePost,
		)
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
