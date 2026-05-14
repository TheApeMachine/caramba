//go:build amd64

package adam

//go:noescape
func adamwStepAVX2(out, m, v, params, grads []float64,
	beta1, oneMinusBeta1, beta2, oneMinusBeta2,
	lrT, eps, lrWD float64)

//go:noescape
func adamwStepSSE2(out, m, v, params, grads []float64,
	beta1, oneMinusBeta1, beta2, oneMinusBeta2,
	lrT, eps, lrWD float64)

func adamwKernel(out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64) {
	if useAVX2 && useFMA {
		adamwStepAVX2(
			out, m, v, params, grads,
			beta1, 1-beta1, beta2, 1-beta2,
			lrT, eps, lrWD,
		)

		return
	}

	adamwStepSSE2(
		out, m, v, params, grads,
		beta1, 1-beta1, beta2, 1-beta2,
		lrT, eps, lrWD,
	)
}
