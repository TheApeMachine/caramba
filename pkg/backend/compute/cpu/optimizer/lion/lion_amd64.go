//go:build amd64

package lion

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func lionStepAVX2(out, m, params, grads []float64,
	lr, beta1, oneMinusBeta1, beta2, oneMinusBeta2, wd float64)

//go:noescape
func lionStepSSE2(out, m, params, grads []float64,
	lr, beta1, oneMinusBeta1, beta2, oneMinusBeta2, wd float64)

func lionStep(out, m, params, grads []float64, lr, beta1, beta2, wd float64) {
	if useAVX2 && useFMA {
		lionStepAVX2(out, m, params, grads, lr, beta1, 1-beta1, beta2, 1-beta2, wd)
		return
	}

	lionStepSSE2(out, m, params, grads, lr, beta1, 1-beta1, beta2, 1-beta2, wd)
}
