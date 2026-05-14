//go:build !amd64 && !arm64

package hebbian

import stdmath "math"

func hebbStep(out, params, grads []float64, lr float64) {
	for index, param := range params {
		out[index] = param + lr*grads[index]
	}
}

func hebbStepNorm(out, params, grads []float64, lr float64) float64 {
	normSq := 0.0

	for index, param := range params {
		value := param + lr*grads[index]
		out[index] = value
		normSq += value * value
	}

	return stdmath.Sqrt(normSq)
}

func hebbScale(out []float64, scale float64) {
	for index := range out {
		out[index] *= scale
	}
}

func ojaStep(out, params, grads []float64, lr, postSq float64) {
	for index, param := range params {
		out[index] = param + lr*(grads[index]-postSq*param)
	}
}

func reduceSumSq(values []float64) float64 {
	sum := 0.0

	for _, value := range values {
		sum += value * value
	}

	return sum
}
