//go:build !amd64 && !arm64

package adagrad

import stdmath "math"

func adagradStep(out, accumulator, params, grads []float64, lr, eps, wd float64) {
	for index, grad := range grads {
		grad += wd * params[index]
		accumulator[index] += grad * grad
		out[index] = params[index] - lr*grad/(stdmath.Sqrt(accumulator[index])+eps)
	}
}

func adadeltaStep(out, eg2, edp2, params, grads []float64, rho, eps, wd float64) {
	oneMinusRho := 1 - rho

	for index, grad := range grads {
		grad += wd * params[index]
		eg2[index] = rho*eg2[index] + oneMinusRho*grad*grad

		update := -stdmath.Sqrt(edp2[index]+eps) / stdmath.Sqrt(eg2[index]+eps) * grad
		edp2[index] = rho*edp2[index] + oneMinusRho*update*update
		out[index] = params[index] + update
	}
}
