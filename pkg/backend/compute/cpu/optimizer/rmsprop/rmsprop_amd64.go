//go:build amd64

package rmsprop

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func rmspropPlainAVX2(out, v, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, wd float64)

//go:noescape
func rmspropPlainSSE2(out, v, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, wd float64)

//go:noescape
func rmspropCenteredAVX2(out, v, gAvg, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, wd float64)

//go:noescape
func rmspropCenteredSSE2(out, v, gAvg, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, wd float64)

//go:noescape
func rmspropMomentumAVX2(out, v, buf, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, momentum, wd float64)

//go:noescape
func rmspropMomentumSSE2(out, v, buf, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, momentum, wd float64)

func rmspropPlain(out, v, params, grads []float64, lr, alpha, eps, wd float64) {
	if useAVX2 && useFMA {
		rmspropPlainAVX2(out, v, params, grads, lr, alpha, 1-alpha, eps, wd)
		return
	}

	rmspropPlainSSE2(out, v, params, grads, lr, alpha, 1-alpha, eps, wd)
}

func rmspropCentered(out, v, gAvg, params, grads []float64, lr, alpha, eps, wd float64) {
	if useAVX2 && useFMA {
		rmspropCenteredAVX2(out, v, gAvg, params, grads, lr, alpha, 1-alpha, eps, wd)
		return
	}

	rmspropCenteredSSE2(out, v, gAvg, params, grads, lr, alpha, 1-alpha, eps, wd)
}

//go:noescape
func rmspropCenteredMomentumAVX2(out, v, gAvg, buf, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, momentum, wd float64)

//go:noescape
func rmspropCenteredMomentumSSE2(out, v, gAvg, buf, params, grads []float64,
	lr, alpha, oneMinusAlpha, eps, momentum, wd float64)

func rmspropCenteredMomentum(out, v, gAvg, buf, params, grads []float64, lr, alpha, eps, momentum, wd float64) {
	if useAVX2 && useFMA {
		rmspropCenteredMomentumAVX2(out, v, gAvg, buf, params, grads, lr, alpha, 1-alpha, eps, momentum, wd)
		return
	}

	rmspropCenteredMomentumSSE2(out, v, gAvg, buf, params, grads, lr, alpha, 1-alpha, eps, momentum, wd)
}

func rmspropMomentum(out, v, buf, params, grads []float64, lr, alpha, eps, momentum, wd float64) {
	if useAVX2 && useFMA {
		rmspropMomentumAVX2(out, v, buf, params, grads, lr, alpha, 1-alpha, eps, momentum, wd)
		return
	}

	rmspropMomentumSSE2(out, v, buf, params, grads, lr, alpha, 1-alpha, eps, momentum, wd)
}
