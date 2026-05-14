//go:build amd64

package vsa

func bundleKernel(dst []float64, inputs [][]float64) {
	for _, input := range inputs {
		if useAVX2 {
			bundleAccumAVX2(dst, input)
		} else {
			bundleAccumSSE2(dst, input)
		}
	}

	if useAVX2 {
		bundleNormalizeAVX2(dst, l2NormEpsilon)
		return
	}

	bundleNormalizeSSE2(dst, l2NormEpsilon)
}
