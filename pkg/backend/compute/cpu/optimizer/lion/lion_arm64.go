//go:build arm64

package lion

//go:noescape
func lionStepNEON(out, m, params, grads []float64,
	lr, beta1, oneMinusBeta1, beta2, oneMinusBeta2, wd float64)

func lionStep(out, m, params, grads []float64, lr, beta1, beta2, wd float64) {
	lionStepNEON(out, m, params, grads, lr, beta1, 1-beta1, beta2, 1-beta2, wd)
}
