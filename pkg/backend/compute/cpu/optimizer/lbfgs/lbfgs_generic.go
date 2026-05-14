//go:build !amd64 && !arm64

package lbfgs

func lbfgsSub(dst, first, second []float64) {
	for index := range first {
		dst[index] = first[index] - second[index]
	}
}

func lbfgsDot(first, second []float64) float64 {
	sum := 0.0

	for index, value := range first {
		sum += value * second[index]
	}

	return sum
}

func lbfgsAddScaled(dst, src []float64, scale float64) {
	for index, value := range src {
		dst[index] += scale * value
	}
}

func lbfgsScale(dst []float64, scale float64) {
	for index := range dst {
		dst[index] *= scale
	}
}

func lbfgsParamStep(out, params, direction []float64, lr float64) {
	for index, param := range params {
		out[index] = param - lr*direction[index]
	}
}
