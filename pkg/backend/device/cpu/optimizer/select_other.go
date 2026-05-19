//go:build !arm64

package optimizer

func adamStepSlices(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	adamStepSlicesScalar(config, params, gradients, firstMoment, secondMoment, output)
}

func adamWStepSlices(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	adamWStepSlicesScalar(config, params, gradients, firstMoment, secondMoment, output)
}

func sgdStepSlices(config SGDConfig, params, gradients, momentum, output []float32) {
	sgdStepSlicesScalar(config, params, gradients, momentum, output)
}

func adamaxStepSlices(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	adamaxStepSlicesScalar(config, params, gradients, firstMoment, infinityMoment, output)
}

func adagradStepSlices(config AdagradConfig, params, gradients, accumulator, output []float32) {
	adagradStepSlicesScalar(config, params, gradients, accumulator, output)
}

func rmspropStepSlices(config RMSpropConfig, params, gradients, secondMoment, output []float32) {
	rmspropStepSlicesScalar(config, params, gradients, secondMoment, output)
}

func lionStepSlices(config LionConfig, params, gradients, momentum, output []float32) {
	lionStepSlicesScalar(config, params, gradients, momentum, output)
}

func larsStepSlices(config LARSConfig, params, gradients, momentum, output []float32) {
	larsStepSlicesScalar(config, params, gradients, momentum, output)
}

func lbfgsStepSlices(config LBFGSConfig, params, gradients, output []float32) {
	lbfgsStepSlicesScalar(config, params, gradients, output)
}

func hebbianStepSlices(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	hebbianStepSlicesScalar(config, weights, post, pre, output, preDim)
}
