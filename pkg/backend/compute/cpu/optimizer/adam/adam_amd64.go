//go:build amd64

package adam

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func adamStepAVX2(
	out, m, v, params, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2, lrT, eps, lrWD float64,
)

//go:noescape
func adamStepSSE2(
	out, m, v, params, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2, lrT, eps, lrWD float64,
)

func adamKernel(
	out, m, v, params, grads []float64, beta1, beta2, lrT, eps, lrWD float64,
) {
	if useAVX2 && useFMA {
		adamStepAVX2(
			out, m, v, params, grads, beta1, 1-beta1, beta2, 1-beta2, lrT, eps, 0,
		)

		return
	}

	adamStepSSE2(
		out, m, v, params, grads, beta1, 1-beta1, beta2, 1-beta2, lrT, eps, 0,
	)
}
