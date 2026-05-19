package math

import (
	stdmath "math"
)

const (
	SeluAlpha      = 1.6732632423543772
	SeluScale      = 1.0507009873554805
	EluAlpha       = 1.0
	CeluAlpha      = 1.0
	LeakyReluSlope = 0.01
	HardTanhMin    = -1.0
	HardTanhMax    = 1.0
	SqrtTwoOverTwo = 0.7071067811865475 // 1/sqrt(2) for GELU erf form
)

func FastReLU64(value float64) float64 {
	if value > 0 {
		return value
	}

	return 0
}

func FastReLU32(value float32) float32 {
	if value > 0 {
		return value
	}

	return 0
}

func FastLeakyReLU64(value float64) float64 {
	if value > 0 {
		return value
	}

	return LeakyReluSlope * value
}

func FastLeakyReLU32(value float32) float32 {
	if value > 0 {
		return value
	}

	return LeakyReluSlope * value
}

func FastGelu64(value float64) float64 {
	return 0.5 * value * (1 + stdmath.Erf(value*SqrtTwoOverTwo))
}

func FastGelu32(value float32) float32 {
	valueFloat64 := float64(value)
	return float32(0.5 * valueFloat64 * (1 + stdmath.Erf(valueFloat64*SqrtTwoOverTwo)))
}

func FastELU64(value float64) float64 {
	if value > 0 {
		return value
	}

	return EluAlpha * (FastExp64(value) - 1)
}

func FastELU32(value float32) float32 {
	if value > 0 {
		return value
	}

	return float32(EluAlpha) * (FastExp32(value) - 1)
}

func FastCELU64(value float64) float64 {
	if value > 0 {
		return value
	}

	return CeluAlpha * (FastExp64(value/CeluAlpha) - 1)
}

func FastCELU32(value float32) float32 {
	if value > 0 {
		return value
	}

	return float32(CeluAlpha) * (FastExp32(value/float32(CeluAlpha)) - 1)
}

func FastSELU64(value float64) float64 {
	if value > 0 {
		return SeluScale * value
	}

	return SeluScale * SeluAlpha * (FastExp64(value) - 1)
}

func FastSELU32(value float32) float32 {
	if value > 0 {
		return float32(SeluScale) * value
	}

	return float32(SeluScale*SeluAlpha) * (FastExp32(value) - 1)
}

func FastSoftplus64(value float64) float64 {
	if value > 20 {
		return value
	}

	return FastLog64(1 + FastExp64(value))
}

func FastSoftplus32(value float32) float32 {
	if value > 20 {
		return value
	}

	return FastLog32(1 + FastExp32(value))
}

func FastMish64(value float64) float64 {
	return value * FastTanh64(FastSoftplus64(value))
}

func FastMish32(value float32) float32 {
	return value * FastTanh32(FastSoftplus32(value))
}

func FastSoftsign64(value float64) float64 {
	return value / (1 + stdmath.Abs(value))
}

func FastSoftsign32(value float32) float32 {
	if value >= 0 {
		return value / (1 + value)
	}

	return value / (1 - value)
}

func FastHardSigmoid64(value float64) float64 {
	shifted := value/6 + 0.5
	if shifted < 0 {
		return 0
	}

	if shifted > 1 {
		return 1
	}

	return shifted
}

func FastHardSigmoid32(value float32) float32 {
	shifted := value/6 + 0.5
	if shifted < 0 {
		return 0
	}

	if shifted > 1 {
		return 1
	}

	return shifted
}

func FastHardSwish64(value float64) float64 {
	return value * FastHardSigmoid64(value+3)
}

func FastHardSwish32(value float32) float32 {
	return value * FastHardSigmoid32(value+3)
}

func FastHardTanh64(value float64) float64 {
	if value < HardTanhMin {
		return HardTanhMin
	}

	if value > HardTanhMax {
		return HardTanhMax
	}

	return value
}

func FastHardTanh32(value float32) float32 {
	if value < HardTanhMin {
		return float32(HardTanhMin)
	}

	if value > HardTanhMax {
		return float32(HardTanhMax)
	}

	return value
}

func FastLog1p64(value float64) float64 {
	return FastLog64(1 + value)
}

func FastLog1p32(value float32) float32 {
	return FastLog32(1 + value)
}

func FastExpm1_64(value float64) float64 {
	return FastExp64(value) - 1
}

func FastExpm1_32(value float32) float32 {
	return FastExp32(value) - 1
}

func FastLogSigmoid64(value float64) float64 {
	return -FastSoftplus64(-value)
}

func FastLogSigmoid32(value float32) float32 {
	return -FastSoftplus32(-value)
}

func FastPReLU32(value, negativeSlope float32) float32 {
	if value > 0 {
		return value
	}

	return negativeSlope * value
}

func FastLeakyReLUWithSlope32(value, negativeSlope float32) float32 {
	if value > 0 {
		return value
	}

	return negativeSlope * value
}

func FastELUWithAlpha32(value, alpha float32) float32 {
	if value > 0 {
		return value
	}

	return alpha * (FastExp32(value) - 1)
}

func FastCELUWithAlpha32(value, alpha float32) float32 {
	if value > 0 {
		return value
	}

	return alpha * (FastExp32(value/alpha) - 1)
}

func FastThreshold32(value, threshold float32) float32 {
	if value > threshold {
		return value
	}

	return 0
}

func FastHardTanhRange32(value, minVal, maxVal float32) float32 {
	if value < minVal {
		return minVal
	}

	if value > maxVal {
		return maxVal
	}

	return value
}

func FastSnake32(value, alpha float32) float32 {
	sine := FastSin32(alpha * value)
	return value + (1/alpha)*sine*sine
}

func FastSnake64(value, alpha float64) float64 {
	return float64(FastSnake32(float32(value), float32(alpha)))
}

func FastGLU32(gate, up float32) float32 {
	return gate * FastSigmoid32(up)
}

func FastGeGLU32(gate, up float32) float32 {
	return gate * FastGelu32(up)
}

func FastSwiGLU32(gate, up float32) float32 {
	return FastSilu32(gate) * up
}

func FastReGLU32(gate, up float32) float32 {
	return gate * FastReLU32(up)
}

func FastSiGLU32(gate, up float32) float32 {
	return FastSigmoid32(gate) * up
}

func FastGeGLUTanh32(gate, up float32) float32 {
	return gate * FastGeluTanh32(up)
}
