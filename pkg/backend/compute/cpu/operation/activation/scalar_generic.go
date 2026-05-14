//go:build !amd64 && !arm64

package activation

import "math"

func scalarTanhKernel(dst, src []float64) {
	for index, value := range src {
		dst[index] = math.Tanh(value)
	}
}

func scalarSigmoidKernel(dst, src []float64) {
	for index, value := range src {
		dst[index] = 1 / (1 + math.Exp(-value))
	}
}

func scalarReLUKernel(dst, src []float64) {
	for index, value := range src {
		if value > 0 {
			dst[index] = value
		}
	}
}

func scalarLeakyReLUKernel(dst, src []float64, alpha float64) {
	for index, value := range src {
		if value < 0 {
			dst[index] = alpha * value

			continue
		}

		dst[index] = value
	}
}

func scalarGeLUKernel(dst, src []float64) {
	for index, value := range src {
		cube := value * value * value
		z := 0.7978845608028654 * (value + 0.044715*cube)
		dst[index] = 0.5 * value * (1 + math.Tanh(z))
	}
}

func scalarSwiGLUKernel(dst, src []float64) {
	half := len(dst)

	for index := range dst {
		gate := src[index]
		value := src[half+index]
		dst[index] = gate / (1 + math.Exp(-gate)) * value
	}
}
