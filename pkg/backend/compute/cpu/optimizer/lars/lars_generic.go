//go:build !amd64 && !arm64

package lars

import stdmath "math"

func larsStep(out, velocity, params, grads []float64, localLR, momentum, wd float64) {
	for index, param := range params {
		grad := grads[index] + wd*param
		velocity[index] = momentum*velocity[index] + localLR*grad
		out[index] = param - velocity[index]
	}
}

func lambEMA(m, v, grads []float64, beta1, beta2 float64) {
	oneMinusBeta1 := 1 - beta1
	oneMinusBeta2 := 1 - beta2

	for index, grad := range grads {
		m[index] = beta1*m[index] + oneMinusBeta1*grad
		v[index] = beta2*v[index] + oneMinusBeta2*grad*grad
	}
}

func lambL2NormSq(values []float64) float64 {
	sum := 0.0

	for _, value := range values {
		sum += value * value
	}

	return sum
}

func lambUpdateNormSq(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64 {
	sum := 0.0

	for index, param := range params {
		update := m[index]*bc1Inv/(stdmath.Sqrt(v[index]*bc2Inv)+eps) + wd*param
		sum += update * update
	}

	return sum
}

func lambStep(out, m, v, params, grads []float64, ratio, bc1Inv, bc2Inv, eps, wd float64) {
	for index, param := range params {
		update := m[index]*bc1Inv/(stdmath.Sqrt(v[index]*bc2Inv)+eps) + wd*param
		out[index] = param - ratio*update
	}
}
