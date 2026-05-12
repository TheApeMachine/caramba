//go:build arm64

package adagrad

//go:noescape
func adagradStepNEON(out, G, params, grads []float64, lr, eps, wd float64)

//go:noescape
func adadeltaStepNEON(out, eg2, edp2, params, grads []float64,
	rho, oneMinusRho, eps, wd float64)

func adagradStep(out, G, params, grads []float64, lr, eps, wd float64) {
	adagradStepNEON(out, G, params, grads, lr, eps, wd)
}

func adadeltaStep(out, eg2, edp2, params, grads []float64, rho, eps, wd float64) {
	adadeltaStepNEON(out, eg2, edp2, params, grads, rho, 1-rho, eps, wd)
}
