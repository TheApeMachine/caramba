package optimizer

import "math"

func SqrtFloat32Native(dst, src []float32) {
	for index := range src {
		dst[index] = float32(math.Sqrt(float64(src[index])))
	}
}
