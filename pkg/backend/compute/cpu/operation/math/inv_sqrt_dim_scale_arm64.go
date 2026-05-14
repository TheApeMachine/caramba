//go:build arm64

package math

func invSqrtDimScaleKernel(out, input []float64, scale float64) {
	copy(out, input)
	mulScalarNEON(out, scale)

	if len(out)%2 != 0 {
		index := len(out) - 1
		out[index] *= scale
	}
}
