package math

//go:noescape
func scalarSqrtTailKernel(dst, src []float64, from int)

//go:noescape
func scalarExpTailKernel(dst, src []float64, from int)

//go:noescape
func scalarLogTailKernel(dst, src []float64, from int)

// scalarSqrtTail fills dst[from:] = sqrt(src[from:]) via a SIMD SQRTSD-based
// per-element kernel — no Go-side math.* calls survive on the hot path.
func scalarSqrtTail(dst, src []float64, from int) {
	scalarSqrtTailKernel(dst, src, from)
}
