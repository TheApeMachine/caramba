package metal

import "math"

var optimizerExpectedUsesFMA = true

func optimizerSGDExpected(params []float32, gradients []float32, state []float32) []float32 {
	out := make([]float32, len(params))

	for index, gradient := range gradients {
		state[index] = optimizerFMA(optimizerSGDMomentum, state[index], gradient)
		out[index] = params[index] - optimizerSGDLR*state[index]
	}

	return out
}

func optimizerLARSExpected(params []float32, gradients []float32, state []float32) []float32 {
	out := make([]float32, len(params))
	paramSum, gradientSum := optimizerLARSGroupedSums(params, gradients)
	paramNorm := float32(math.Sqrt(float64(paramSum)))
	gradientNorm := float32(math.Sqrt(float64(gradientSum)))
	trust := float32(1)

	if paramNorm > 0 && gradientNorm > 0 {
		trust = optimizerLARSTrust * paramNorm /
			(gradientNorm + optimizerLARSDecay*paramNorm + optimizerLARSEps)
	}

	for index, gradient := range gradients {
		decayed := optimizerFMA(optimizerLARSDecay, params[index], gradient)
		state[index] = optimizerFMA(optimizerSGDMomentum, state[index], decayed)
		out[index] = params[index] - optimizerLARSLR*trust*state[index]
	}

	return out
}

func optimizerLARSGroupedSums(params []float32, gradients []float32) (float32, float32) {
	var paramSum float32
	var gradientSum float32

	for groupStart := 0; groupStart < len(params); groupStart += 256 {
		groupEnd := min(groupStart+256, len(params))
		groupParam, groupGradient := optimizerLARSGroupSum(params, gradients, groupStart, groupEnd)
		paramSum += groupParam
		gradientSum += groupGradient
	}

	return paramSum, gradientSum
}

func optimizerLARSGroupSum(
	params []float32,
	gradients []float32,
	groupStart int,
	groupEnd int,
) (float32, float32) {
	paramSums := make([]float32, 256)
	gradientSums := make([]float32, 256)

	for localIndex := range 256 {
		index := groupStart + localIndex
		if index >= groupEnd {
			continue
		}

		paramSums[localIndex] = params[index] * params[index]
		gradientSums[localIndex] = gradients[index] * gradients[index]
	}

	for stride := 128; stride > 0; stride >>= 1 {
		for localIndex := range stride {
			paramSums[localIndex] += paramSums[localIndex+stride]
			gradientSums[localIndex] += gradientSums[localIndex+stride]
		}
	}

	return paramSums[0], gradientSums[0]
}

func optimizerLBFGSExpected(params []float32, gradients []float32) []float32 {
	out := make([]float32, len(params))

	for index, gradient := range gradients {
		out[index] = params[index] - gradient
	}

	return out
}

func optimizerHebbianExpected(weights []float32, post []float32, pre []float32) []float32 {
	out := make([]float32, len(weights))
	preCount := len(pre)

	for postIndex, postValue := range post {
		for preIndex, preValue := range pre {
			weightIndex := postIndex*preCount + preIndex
			update := optimizerHebbianLR * postValue * preValue
			out[weightIndex] = optimizerFMA(weights[weightIndex], 1-optimizerHebbianDec, update)
		}
	}

	return out
}

func optimizerSign(value float32) float32 {
	if value > 0 {
		return 1
	}

	if value < 0 {
		return -1
	}

	return 0
}

func optimizerFMA(left float32, right float32, addend float32) float32 {
	if !optimizerExpectedUsesFMA {
		return left*right + addend
	}

	return float32(math.FMA(float64(left), float64(right), float64(addend)))
}
