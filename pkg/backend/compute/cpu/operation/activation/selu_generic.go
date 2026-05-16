//go:build !amd64 && !arm64

package activation

import "math"

func seluKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic("SELU: dst shorter than src")
	}

	seluBlendGeneric(dst, src, nil)
}

func seluBlend(dst, src, expValues []float64) {
	seluBlendGeneric(dst, src, expValues)
}

func seluBlendGeneric(dst, src, expValues []float64) {
	for index, value := range src {
		if value > 0 {
			dst[index] = seluScale * value

			continue
		}

		expValue := math.Exp(value)

		if len(expValues) > index {
			expValue = expValues[index]
		}

		dst[index] = seluScaleAlpha * (expValue - 1)
	}
}
