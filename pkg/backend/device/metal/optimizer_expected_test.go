package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

const (
	optimizerAdamLR      = float32(1.0e-4)
	optimizerAdamBeta1   = float32(0.9)
	optimizerAdamBeta2   = float32(0.999)
	optimizerAdamEpsilon = float32(1.0e-8)
	optimizerAdamWDecay  = float32(1.0e-2)
	optimizerAdamaxLR    = float32(2.0e-3)
	optimizerAdagradLR   = float32(1.0e-2)
	optimizerAdagradEps  = float32(1.0e-10)
	optimizerRMSpropLR   = float32(1.0e-3)
	optimizerRMSpropRate = float32(0.99)
	optimizerLionLR      = float32(1.0e-4)
	optimizerLionBeta2   = float32(0.99)
	optimizerSGDLR       = float32(1.0e-2)
	optimizerSGDMomentum = float32(0.9)
	optimizerLARSLR      = float32(1.0e-2)
	optimizerLARSDecay   = float32(1.0e-4)
	optimizerLARSTrust   = float32(1.0e-3)
	optimizerLARSEps     = float32(1.0e-8)
	optimizerHebbianLR   = float32(1.0e-3)
	optimizerHebbianDec  = float32(1.0e-4)
)

func optimizerStorageInputs(
	elementCount int,
	storageDType dtype.DType,
) ([]byte, []byte, []float32, []float32) {
	paramValues := projectionValues(elementCount, 67, 64)
	gradientValues := projectionValues(elementCount, 71, 128)
	paramBytes := encodeProjectionValuesAsDType(paramValues, storageDType)
	gradientBytes := encodeProjectionValuesAsDType(gradientValues, storageDType)

	return paramBytes, gradientBytes,
		decodeDTypeBytesToFloat32(paramBytes, storageDType),
		decodeDTypeBytesToFloat32(gradientBytes, storageDType)
}

func optimizerStateValues(elementCount int, seed int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32((index*seed)%29+1) / 512
	}

	return values
}

func optimizerStateBytes(values []float32) []byte {
	return dtypeconvert.Float32ToBytes(values)
}

func optimizerAdamExpected(
	params []float32,
	gradients []float32,
	first []float32,
	second []float32,
) []float32 {
	out := make([]float32, len(params))
	firstCorrection := 1 - optimizerAdamBeta1
	secondCorrection := 1 - optimizerAdamBeta2

	for index, gradient := range gradients {
		first[index] = optimizerFMA(optimizerAdamBeta1, first[index], (1-optimizerAdamBeta1)*gradient)
		second[index] = optimizerFMA(
			optimizerAdamBeta2, second[index], (1-optimizerAdamBeta2)*gradient*gradient,
		)
		correctedFirst := first[index] / firstCorrection
		correctedSecond := second[index] / secondCorrection
		denominator := float32(math.Sqrt(float64(correctedSecond))) + optimizerAdamEpsilon
		out[index] = params[index] - optimizerAdamLR*correctedFirst/denominator
	}

	return out
}

func optimizerAdamWExpected(
	params []float32,
	gradients []float32,
	first []float32,
	second []float32,
) []float32 {
	out := make([]float32, len(params))
	firstCorrection := 1 - optimizerAdamBeta1
	secondCorrection := 1 - optimizerAdamBeta2

	for index, gradient := range gradients {
		first[index] = optimizerFMA(optimizerAdamBeta1, first[index], (1-optimizerAdamBeta1)*gradient)
		second[index] = optimizerFMA(
			optimizerAdamBeta2, second[index], (1-optimizerAdamBeta2)*gradient*gradient,
		)
		correctedFirst := first[index] / firstCorrection
		correctedSecond := second[index] / secondCorrection
		denominator := float32(math.Sqrt(float64(correctedSecond))) + optimizerAdamEpsilon
		gradientStep := optimizerAdamLR * correctedFirst / denominator
		decayStep := optimizerAdamLR * optimizerAdamWDecay * params[index]
		out[index] = params[index] - gradientStep - decayStep
	}

	return out
}

func optimizerAdamaxExpected(
	params []float32,
	gradients []float32,
	first []float32,
	infinity []float32,
) []float32 {
	out := make([]float32, len(params))
	firstCorrection := 1 - optimizerAdamBeta1

	for index, gradient := range gradients {
		first[index] = optimizerFMA(optimizerAdamBeta1, first[index], (1-optimizerAdamBeta1)*gradient)
		infinity[index] = max(optimizerAdamBeta2*infinity[index], float32(math.Abs(float64(gradient))))
		correctedFirst := first[index] / firstCorrection
		out[index] = params[index] -
			optimizerAdamaxLR*correctedFirst/(infinity[index]+optimizerAdamEpsilon)
	}

	return out
}

func optimizerAdagradExpected(
	params []float32,
	gradients []float32,
	state []float32,
) []float32 {
	out := make([]float32, len(params))

	for index, gradient := range gradients {
		state[index] += gradient * gradient
		denominator := float32(math.Sqrt(float64(state[index]))) + optimizerAdagradEps
		out[index] = params[index] - optimizerAdagradLR*gradient/denominator
	}

	return out
}

func optimizerRMSpropExpected(
	params []float32,
	gradients []float32,
	state []float32,
) []float32 {
	out := make([]float32, len(params))

	for index, gradient := range gradients {
		state[index] = optimizerFMA(
			optimizerRMSpropRate, state[index], (1-optimizerRMSpropRate)*gradient*gradient,
		)
		denominator := float32(math.Sqrt(float64(state[index]))) + optimizerAdamEpsilon
		out[index] = params[index] - optimizerRMSpropLR*gradient/denominator
	}

	return out
}

func optimizerLionExpected(params []float32, gradients []float32, state []float32) []float32 {
	out := make([]float32, len(params))

	for index, gradient := range gradients {
		update := optimizerFMA(optimizerAdamBeta1, state[index], (1-optimizerAdamBeta1)*gradient)
		out[index] = params[index] - optimizerLionLR*optimizerSign(update)
		state[index] = optimizerFMA(optimizerLionBeta2, state[index], (1-optimizerLionBeta2)*gradient)
	}

	return out
}
