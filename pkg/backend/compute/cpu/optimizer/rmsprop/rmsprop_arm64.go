//go:build arm64

package rmsprop

//go:noescape
func rmspropPlainNEON(out, v, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, wd float64)

//go:noescape
func rmspropCenteredNEON(out, v, gAvg, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, wd float64)

//go:noescape
func rmspropMomentumNEON(out, v, buf, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, momentum, wd float64)

func rmspropPlain(out, v, params, grads []float64, lr, alpha, eps, wd float64) {
	rmspropPlainNEON(out, v, params, grads, lr, alpha, 1-alpha, eps, wd)
}

func rmspropCentered(out, v, gAvg, params, grads []float64, lr, alpha, eps, wd float64) {
	rmspropCenteredNEON(out, v, gAvg, params, grads, lr, alpha, 1-alpha, eps, wd)
}

//go:noescape
func rmspropCenteredMomentumNEON(out, v, gAvg, buf, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, momentum, wd float64)

func rmspropCenteredMomentum(out, v, gAvg, buf, params, grads []float64, lr, alpha, eps, momentum, wd float64) {
	rmspropCenteredMomentumNEON(out, v, gAvg, buf, params, grads, lr, alpha, 1-alpha, eps, momentum, wd)
}

func rmspropMomentum(out, v, buf, params, grads []float64, lr, alpha, eps, momentum, wd float64) {
	rmspropMomentumNEON(out, v, buf, params, grads, lr, alpha, 1-alpha, eps, momentum, wd)
}
