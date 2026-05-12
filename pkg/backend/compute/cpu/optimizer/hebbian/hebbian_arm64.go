//go:build arm64

package hebbian

//go:noescape
func hebbStepNEON(out, params, grads []float64, lr float64)

//go:noescape
func hebbStepNormNEON(out, params, grads []float64, lr float64) float64

//go:noescape
func hebbScaleNEON(out []float64, scale float64)

//go:noescape
func ojaStepNEON(out, params, grads []float64, lr, postSq float64)

//go:noescape
func reduceSumSqNEON(a []float64) float64

func hebbStep(out, params, grads []float64, lr float64) {
	hebbStepNEON(out, params, grads, lr)
}

func hebbStepNorm(out, params, grads []float64, lr float64) float64 {
	return hebbStepNormNEON(out, params, grads, lr)
}

func hebbScale(out []float64, scale float64) {
	hebbScaleNEON(out, scale)
}

func ojaStep(out, params, grads []float64, lr, postSq float64) {
	ojaStepNEON(out, params, grads, lr, postSq)
}

func reduceSumSq(a []float64) float64 {
	return reduceSumSqNEON(a)
}
