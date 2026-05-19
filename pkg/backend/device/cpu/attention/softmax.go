package attention

import "math"

func SoftmaxRowFillExpsNative(dst, src []float32, maximum float32) float32 {
	var sum float32

	for index, value := range src {
		shifted := float32(math.Exp(float64(value - maximum)))
		dst[index] = shifted
		sum += shifted
	}

	return sum
}
