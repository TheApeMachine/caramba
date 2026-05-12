//go:build arm64

package adam

import stdmath "math"

//go:noescape
func adamStepNEON(out, m, v, params, grads []float64,
	beta1, oneMinusBeta1, beta2, oneMinusBeta2,
	lrT, eps, lrWD float64)

func adamStep(out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64) {
	adamStepNEON(out, m, v, params, grads,
		beta1, 1-beta1, beta2, 1-beta2, lrT, eps, lrWD)
}

//go:noescape
func adamaxStepNEON(out, m, u, params, grads []float64,
	beta1, oneMinusBeta1, beta2, lrT, eps float64)

func adamaxStep(out, m, u, params, grads []float64, beta1, beta2, lrT, eps float64) {
	adamaxStepNEON(out, m, u, params, grads, beta1, 1-beta1, beta2, lrT, eps)
}

func biasCorrectedLR(lr, beta1, beta2 float64, step int) float64 {
	bc1 := 1 - stdmath.Pow(beta1, float64(step))
	bc2 := 1 - stdmath.Pow(beta2, float64(step))

	return lr * stdmath.Sqrt(bc2) / bc1
}
