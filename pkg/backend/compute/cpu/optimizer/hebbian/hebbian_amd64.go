//go:build amd64

package hebbian

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func hebbStepAVX2(out, params, grads []float64, lr float64)

//go:noescape
func hebbStepSSE2(out, params, grads []float64, lr float64)

//go:noescape
func hebbStepNormAVX2(out, params, grads []float64, lr float64) float64

//go:noescape
func hebbStepNormSSE2(out, params, grads []float64, lr float64) float64

//go:noescape
func hebbScaleAVX2(out []float64, scale float64)

//go:noescape
func hebbScaleSSE2(out []float64, scale float64)

//go:noescape
func ojaStepAVX2(out, params, grads []float64, lr, postSq float64)

//go:noescape
func ojaStepSSE2(out, params, grads []float64, lr, postSq float64)

//go:noescape
func reduceSumSqAVX2(a []float64) float64

//go:noescape
func reduceSumSqSSE2(a []float64) float64

func hebbStep(out, params, grads []float64, lr float64) {
	if useAVX2 && useFMA {
		hebbStepAVX2(out, params, grads, lr)
		return
	}

	hebbStepSSE2(out, params, grads, lr)
}

func hebbStepNorm(out, params, grads []float64, lr float64) float64 {
	if useAVX2 && useFMA {
		return hebbStepNormAVX2(out, params, grads, lr)
	}

	return hebbStepNormSSE2(out, params, grads, lr)
}

func hebbScale(out []float64, scale float64) {
	if useAVX2 {
		hebbScaleAVX2(out, scale)
		return
	}

	hebbScaleSSE2(out, scale)
}

func ojaStep(out, params, grads []float64, lr, postSq float64) {
	if useAVX2 && useFMA {
		ojaStepAVX2(out, params, grads, lr, postSq)
		return
	}

	ojaStepSSE2(out, params, grads, lr, postSq)
}

func reduceSumSq(a []float64) float64 {
	if useAVX2 && useFMA {
		return reduceSumSqAVX2(a)
	}

	return reduceSumSqSSE2(a)
}
