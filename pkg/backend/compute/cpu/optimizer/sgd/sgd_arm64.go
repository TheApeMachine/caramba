//go:build arm64

package sgd

//go:noescape
func sgdVanillaNEON(out, params, grads []float64, lr, wd float64)

//go:noescape
func sgdMomentumNEON(out, params, grads, velocity []float64,
	lr, wd, momentum float64, nesterov uint64)

func sgdVanilla(out, params, grads []float64, lr, wd float64) {
	sgdVanillaNEON(out, params, grads, lr, wd)
}

func sgdMomentum(out, params, grads, velocity []float64, lr, wd, momentum float64, nesterov bool) {
	flag := uint64(0)
	if nesterov {
		flag = 1
	}

	sgdMomentumNEON(out, params, grads, velocity, lr, wd, momentum, flag)
}
