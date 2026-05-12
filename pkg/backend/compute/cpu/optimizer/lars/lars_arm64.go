//go:build arm64

package lars

//go:noescape
func larsStepNEON(out, velocity, params, grads []float64, localLR, momentum, wd float64)

//go:noescape
func lambEMANEON(m, v, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2 float64)

//go:noescape
func lambL2NormSqNEON(a []float64) float64

//go:noescape
func lambUpdateNormSqNEON(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64

//go:noescape
func lambStepNEON(out, m, v, params, grads []float64, ratio, bc1Inv, bc2Inv, eps, wd float64)

func larsStep(out, velocity, params, grads []float64, localLR, momentum, wd float64) {
	larsStepNEON(out, velocity, params, grads, localLR, momentum, wd)
}

func lambEMA(m, v, grads []float64, beta1, beta2 float64) {
	lambEMANEON(m, v, grads, beta1, 1-beta1, beta2, 1-beta2)
}

func lambL2NormSq(a []float64) float64 {
	return lambL2NormSqNEON(a)
}

func lambUpdateNormSq(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64 {
	return lambUpdateNormSqNEON(m, v, params, bc1Inv, bc2Inv, eps, wd)
}

func lambStep(out, m, v, params, grads []float64, ratio, bc1Inv, bc2Inv, eps, wd float64) {
	lambStepNEON(out, m, v, params, grads, ratio, bc1Inv, bc2Inv, eps, wd)
}
