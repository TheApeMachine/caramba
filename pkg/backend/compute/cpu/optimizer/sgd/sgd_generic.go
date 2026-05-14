//go:build !amd64 && !arm64

package sgd

func sgdVanilla(out, params, grads []float64, lr, wd float64) {
	for index, param := range params {
		grad := grads[index] + wd*param
		out[index] = param - lr*grad
	}
}

func sgdMomentum(
	out, params, grads, velocity []float64, lr, wd, momentum float64, nesterov bool,
) {
	for index, param := range params {
		grad := grads[index] + wd*param
		velocity[index] = momentum*velocity[index] + grad

		if nesterov {
			out[index] = param - lr*(grad+momentum*velocity[index])
			continue
		}

		out[index] = param - lr*velocity[index]
	}
}
