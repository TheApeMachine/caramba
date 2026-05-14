//go:build !amd64 && !arm64

package adam

import stdmath "math"

func adamKernel(out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64) {
	for index := range params {
		g := grads[index]
		m[index] = beta1*m[index] + (1-beta1)*g
		v[index] = beta2*v[index] + (1-beta2)*g*g
		out[index] = params[index] - lrT*m[index]/(stdmath.Sqrt(v[index])+eps)
	}
}
