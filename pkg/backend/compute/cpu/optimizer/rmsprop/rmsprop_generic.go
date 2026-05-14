//go:build !amd64 && !arm64

package rmsprop

import stdmath "math"

func rmspropPlain(out, v, params, grads []float64, lr, alpha, eps, wd float64) {
	oneMinusAlpha := 1 - alpha

	for index, param := range params {
		grad := grads[index] + wd*param
		v[index] = alpha*v[index] + oneMinusAlpha*grad*grad
		out[index] = param - lr*grad/(stdmath.Sqrt(v[index])+eps)
	}
}

func rmspropCentered(out, v, gradAvg, params, grads []float64, lr, alpha, eps, wd float64) {
	oneMinusAlpha := 1 - alpha

	for index, param := range params {
		grad := grads[index] + wd*param
		v[index] = alpha*v[index] + oneMinusAlpha*grad*grad
		gradAvg[index] = alpha*gradAvg[index] + oneMinusAlpha*grad

		centered := v[index] - gradAvg[index]*gradAvg[index]
		out[index] = param - lr*grad/(stdmath.Sqrt(centered)+eps)
	}
}

func rmspropMomentum(
	out, v, buf, params, grads []float64, lr, alpha, eps, momentum, wd float64,
) {
	oneMinusAlpha := 1 - alpha

	for index, param := range params {
		grad := grads[index] + wd*param
		v[index] = alpha*v[index] + oneMinusAlpha*grad*grad
		buf[index] = momentum*buf[index] + grad/(stdmath.Sqrt(v[index])+eps)
		out[index] = param - lr*buf[index]
	}
}

func rmspropCenteredMomentum(
	out, v, gradAvg, buf, params, grads []float64, lr, alpha, eps, momentum, wd float64,
) {
	oneMinusAlpha := 1 - alpha

	for index, param := range params {
		grad := grads[index] + wd*param
		v[index] = alpha*v[index] + oneMinusAlpha*grad*grad
		gradAvg[index] = alpha*gradAvg[index] + oneMinusAlpha*grad

		centered := v[index] - gradAvg[index]*gradAvg[index]
		buf[index] = momentum*buf[index] + grad/(stdmath.Sqrt(centered)+eps)
		out[index] = param - lr*buf[index]
	}
}
