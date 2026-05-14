//go:build !amd64 && !arm64

package adam

import stdmath "math"

func adamwKernel(out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64) {
	oneMinusBeta1 := 1 - beta1
	oneMinusBeta2 := 1 - beta2

	for index, grad := range grads {
		m[index] = beta1*m[index] + oneMinusBeta1*grad
		v[index] = beta2*v[index] + oneMinusBeta2*grad*grad
		out[index] = params[index]*(1-lrWD) - lrT*m[index]/(stdmath.Sqrt(v[index])+eps)
	}
}
