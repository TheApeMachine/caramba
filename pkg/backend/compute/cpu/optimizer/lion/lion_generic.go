//go:build !amd64 && !arm64

package lion

func lionStep(out, m, params, grads []float64, lr, beta1, beta2, wd float64) {
	oneMinusBeta1 := 1 - beta1
	oneMinusBeta2 := 1 - beta2

	for index, param := range params {
		update := beta1*m[index] + oneMinusBeta1*grads[index]
		out[index] = param - lr*(sign(update)+wd*param)
		m[index] = beta2*m[index] + oneMinusBeta2*grads[index]
	}
}

func sign(value float64) float64 {
	if value > 0 {
		return 1
	}

	if value < 0 {
		return -1
	}

	return 0
}
