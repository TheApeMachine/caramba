//go:build amd64

package cpu

import "math"

/*
amd64 dispatcher for unary float32 abs/neg/sqrt. SSE2 has SQRTPS,
the bit-mask tricks for ABS (AND with 0x7FFFFFFF mask) and NEG
(XOR with 0x80000000 mask) land in .s files in a hardware-verified
session; today this routes through the scalar reference.
*/

func AbsFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = float32(math.Abs(float64(value)))
	}
}

func NegFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = -value
	}
}

func SqrtFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = float32(math.Sqrt(float64(value)))
	}
}

func ReluFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = 0

		if value > 0 {
			dst[index] = value
		}
	}
}
