package math

import stdmath "math"

// scalarSqrtTail fills dst[from:] = sqrt(src[from:]).
// Called by SIMD sqrtVec to handle non-aligned remainders.
func scalarSqrtTail(dst, src []float64, from int) {
	for i := from; i < len(src); i++ {
		dst[i] = stdmath.Sqrt(src[i])
	}
}
