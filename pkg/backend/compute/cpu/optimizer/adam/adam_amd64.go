//go:build amd64

package adam

import (
	stdmath "math"

	"golang.org/x/sys/cpu"
)

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func adamStepAVX2(out, m, v, params, grads []float64,
	beta1, oneMinusBeta1, beta2, oneMinusBeta2,
	lrT, eps, lrWD float64)

//go:noescape
func adamStepSSE2(out, m, v, params, grads []float64,
	beta1, oneMinusBeta1, beta2, oneMinusBeta2,
	lrT, eps, lrWD float64)

//go:noescape
func adamaxStepAVX2(out, m, u, params, grads []float64,
	beta1, oneMinusBeta1, beta2, lrT, eps float64)

//go:noescape
func adamaxStepSSE2(out, m, u, params, grads []float64,
	beta1, oneMinusBeta1, beta2, lrT, eps float64)

func adamaxStep(out, m, u, params, grads []float64, beta1, beta2, lrT, eps float64) {
	if useAVX2 && useFMA {
		adamaxStepAVX2(out, m, u, params, grads, beta1, 1-beta1, beta2, lrT, eps)
		return
	}

	adamaxStepSSE2(out, m, u, params, grads, beta1, 1-beta1, beta2, lrT, eps)
}

func adamStep(out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64) {
	if useAVX2 && useFMA {
		adamStepAVX2(out, m, v, params, grads,
			beta1, 1-beta1, beta2, 1-beta2, lrT, eps, lrWD)
		return
	}

	adamStepSSE2(out, m, v, params, grads,
		beta1, 1-beta1, beta2, 1-beta2, lrT, eps, lrWD)
}

func biasCorrectedLR(lr, beta1, beta2 float64, step int) float64 {
	bc1 := 1 - stdmath.Pow(beta1, float64(step))
	bc2 := 1 - stdmath.Pow(beta2, float64(step))

	return lr * stdmath.Sqrt(bc2) / bc1
}
