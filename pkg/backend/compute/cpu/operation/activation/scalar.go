package activation

import (
	"fmt"
	"math"
)

func scalarGeLU(dst, src []float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	for index := 0; index < elementCount; index++ {
		value := src[index]
		cube := value * value * value
		z := 0.7978845608028654 * (value + 0.044715*cube)
		dst[index] = 0.5 * value * (1 + math.Tanh(z))
	}
}

func scalarLeakyReLU(dst, src []float64, alpha float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	for index := 0; index < elementCount; index++ {
		value := src[index]

		if value < 0 {
			dst[index] = alpha * value
		} else {
			dst[index] = value
		}
	}
}

func scalarReLU(dst, src []float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	for index := 0; index < elementCount; index++ {
		value := src[index]

		if value > 0 {
			dst[index] = value
		} else {
			dst[index] = 0
		}
	}
}

func scalarSigmoid(dst, src []float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	for index := 0; index < elementCount; index++ {
		dst[index] = scalarSigmoidAt(src[index])
	}
}

func scalarSigmoidAt(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func scalarSwiGLU(dst, src []float64) {
	half := len(dst)

	if len(src) < 2*half {
		panic(fmt.Sprintf(
			"scalarSwiGLU: src too short for gates|values layout (len(dst)=%d len(src)=%d, need %d)",
			half, len(src), 2*half,
		))
	}

	for index := range dst {
		gate := src[index]
		value := src[half+index]
		dst[index] = scalarSigmoidAt(gate) * value
	}
}

func scalarTanh(dst, src []float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	for index := 0; index < elementCount; index++ {
		dst[index] = math.Tanh(src[index])
	}
}
