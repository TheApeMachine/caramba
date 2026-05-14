//go:build !amd64 && !arm64

package adam

import stdmath "math"

func adamaxKernel(out, m, u, params, grads []float64, beta1, beta2, lrT, eps float64) {
	oneMinusBeta1 := 1 - beta1

	for index, grad := range grads {
		m[index] = beta1*m[index] + oneMinusBeta1*grad
		u[index] = stdmath.Max(beta2*u[index], stdmath.Abs(grad))
		out[index] = params[index] - lrT*m[index]/(u[index]+eps)
	}
}
