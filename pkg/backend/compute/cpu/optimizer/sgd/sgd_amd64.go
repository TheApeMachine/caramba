//go:build amd64

package sgd

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func sgdVanillaAVX2(out, params, grads []float64, lr, wd float64)

//go:noescape
func sgdVanillaSSE2(out, params, grads []float64, lr, wd float64)

//go:noescape
func sgdMomentumAVX2(out, params, grads, velocity []float64,
	lr, wd, momentum float64, nesterov uint64)

//go:noescape
func sgdMomentumSSE2(out, params, grads, velocity []float64,
	lr, wd, momentum float64, nesterov uint64)

func sgdVanilla(out, params, grads []float64, lr, wd float64) {
	if useAVX2 && useFMA {
		sgdVanillaAVX2(out, params, grads, lr, wd)
		return
	}

	sgdVanillaSSE2(out, params, grads, lr, wd)
}

func sgdMomentum(out, params, grads, velocity []float64, lr, wd, momentum float64, nesterov bool) {
	flag := uint64(0)
	if nesterov {
		flag = 1
	}

	if useAVX2 && useFMA {
		sgdMomentumAVX2(out, params, grads, velocity, lr, wd, momentum, flag)
		return
	}

	sgdMomentumSSE2(out, params, grads, velocity, lr, wd, momentum, flag)
}
