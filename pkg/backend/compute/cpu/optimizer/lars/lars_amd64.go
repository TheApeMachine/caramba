//go:build amd64

package lars

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func larsStepAVX2(out, velocity, params, grads []float64, localLR, momentum, wd float64)

//go:noescape
func larsStepSSE2(out, velocity, params, grads []float64, localLR, momentum, wd float64)

//go:noescape
func lambEMAAVX2(m, v, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2 float64)

//go:noescape
func lambEMASSE2(m, v, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2 float64)

//go:noescape
func lambL2NormSqAVX2(a []float64) float64

//go:noescape
func lambL2NormSqSSE2(a []float64) float64

//go:noescape
func lambUpdateNormSqAVX2(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64

//go:noescape
func lambUpdateNormSqSSE2(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64

//go:noescape
func lambStepAVX2(out, m, v, params, grads []float64, ratio, bc1Inv, bc2Inv, eps, wd float64)

//go:noescape
func lambStepSSE2(out, m, v, params, grads []float64, ratio, bc1Inv, bc2Inv, eps, wd float64)

func larsStep(out, velocity, params, grads []float64, localLR, momentum, wd float64) {
	if useAVX2 && useFMA {
		larsStepAVX2(out, velocity, params, grads, localLR, momentum, wd)
		return
	}

	larsStepSSE2(out, velocity, params, grads, localLR, momentum, wd)
}

func lambEMA(m, v, grads []float64, beta1, beta2 float64) {
	if useAVX2 && useFMA {
		lambEMAAVX2(m, v, grads, beta1, 1-beta1, beta2, 1-beta2)
		return
	}

	lambEMASSE2(m, v, grads, beta1, 1-beta1, beta2, 1-beta2)
}

func lambL2NormSq(a []float64) float64 {
	if useAVX2 && useFMA {
		return lambL2NormSqAVX2(a)
	}

	return lambL2NormSqSSE2(a)
}

func lambUpdateNormSq(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64 {
	if useAVX2 && useFMA {
		return lambUpdateNormSqAVX2(m, v, params, bc1Inv, bc2Inv, eps, wd)
	}

	return lambUpdateNormSqSSE2(m, v, params, bc1Inv, bc2Inv, eps, wd)
}

func lambStep(out, m, v, params, grads []float64, ratio, bc1Inv, bc2Inv, eps, wd float64) {
	if useAVX2 && useFMA {
		lambStepAVX2(out, m, v, params, grads, ratio, bc1Inv, bc2Inv, eps, wd)
		return
	}

	lambStepSSE2(out, m, v, params, grads, ratio, bc1Inv, bc2Inv, eps, wd)
}
