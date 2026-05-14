//go:build amd64

package adam

//go:noescape
func adamaxStepAVX2(out, m, u, params, grads []float64,
	beta1, oneMinusBeta1, beta2, lrT, eps float64)

//go:noescape
func adamaxStepSSE2(out, m, u, params, grads []float64,
	beta1, oneMinusBeta1, beta2, lrT, eps float64)

func adamaxKernel(out, m, u, params, grads []float64, beta1, beta2, lrT, eps float64) {
	if useAVX2 && useFMA {
		adamaxStepAVX2(out, m, u, params, grads, beta1, 1-beta1, beta2, lrT, eps)

		return
	}

	adamaxStepSSE2(out, m, u, params, grads, beta1, 1-beta1, beta2, lrT, eps)
}
