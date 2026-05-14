//go:build !amd64 && !arm64

package math

func invSqrtDimScaleKernel(out, input []float64, scale float64) {
	for index, value := range input {
		out[index] = value * scale
	}
}
