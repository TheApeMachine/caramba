//go:build arm64

package vsa

func bundleKernel(dst []float64, inputs [][]float64) {
	for _, input := range inputs {
		bundleAccumNEON(dst, input)
	}

	bundleNormalizeNEON(dst, l2NormEpsilon)
}
