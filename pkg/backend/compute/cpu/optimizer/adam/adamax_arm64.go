//go:build arm64

package adam

//go:noescape
func adamaxStepNEON(out, m, u, params, grads []float64,
	beta1, oneMinusBeta1, beta2, lrT, eps float64)

func adamaxKernel(out, m, u, params, grads []float64, beta1, beta2, lrT, eps float64) {
	adamaxStepNEON(out, m, u, params, grads, beta1, 1-beta1, beta2, lrT, eps)
}
