//go:build arm64

package adam

//go:noescape
func adamStepNEON(out, m, v, params, grads []float64,
	beta1, oneMinusBeta1, beta2, oneMinusBeta2,
	lrT, eps, lrWD float64)

func adamKernel(out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64) {
	adamStepNEON(
		out, m, v, params, grads,
		beta1, 1-beta1, beta2, 1-beta2,
		lrT, eps, lrWD,
	)
}
