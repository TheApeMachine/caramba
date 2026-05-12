//go:build amd64

package adagrad

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func adagradStepAVX2(out, G, params, grads []float64, lr, eps, wd float64)

//go:noescape
func adagradStepSSE2(out, G, params, grads []float64, lr, eps, wd float64)

//go:noescape
func adadeltaStepAVX2(out, eg2, edp2, params, grads []float64,
	rho, oneMinusRho, eps, wd float64)

//go:noescape
func adadeltaStepSSE2(out, eg2, edp2, params, grads []float64,
	rho, oneMinusRho, eps, wd float64)

func adagradStep(out, G, params, grads []float64, lr, eps, wd float64) {
	if useAVX2 && useFMA {
		adagradStepAVX2(out, G, params, grads, lr, eps, wd)
		return
	}

	adagradStepSSE2(out, G, params, grads, lr, eps, wd)
}

func adadeltaStep(out, eg2, edp2, params, grads []float64, rho, eps, wd float64) {
	if useAVX2 && useFMA {
		adadeltaStepAVX2(out, eg2, edp2, params, grads, rho, 1-rho, eps, wd)
		return
	}

	adadeltaStepSSE2(out, eg2, edp2, params, grads, rho, 1-rho, eps, wd)
}
