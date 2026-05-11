//go:build amd64

package causal

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func matVecAVX2(dst, w, x []float64, rows, cols int)

//go:noescape
func matVecSSE2(dst, w, x []float64, rows, cols int)

//go:noescape
func axpyAVX2(dst, src []float64, scale float64)

//go:noescape
func axpySSE2(dst, src []float64, scale float64)

//go:noescape
func dotAVX2(a, b []float64) float64

//go:noescape
func dotSSE2(a, b []float64) float64

//go:noescape
func subVecAVX2(dst, a, b []float64)

//go:noescape
func subVecSSE2(dst, a, b []float64)

/*
applyMatVec computes dst = W @ x where W is [rows x cols] row-major and x is [cols].
*/
func applyMatVec(dst, w, x []float64, rows, cols int) {
	if useAVX2 && useFMA {
		matVecAVX2(dst, w, x, rows, cols)
	} else {
		matVecSSE2(dst, w, x, rows, cols)
	}
}

/*
applyAxpy computes dst += scale * src elementwise.
*/
func applyAxpy(dst, src []float64, scale float64) {
	limit := alignedLen(len(src))

	if useAVX2 {
		if limit > 0 {
			axpyAVX2(dst[:limit], src[:limit], scale)
		}
	} else if limit > 0 {
		axpySSE2(dst[:limit], src[:limit], scale)
	}

	for idx := limit; idx < len(src); idx++ {
		dst[idx] += scale * src[idx]
	}
}

/*
applyDotProduct computes the inner product of a and b.
*/
func applyDotProduct(a, b []float64) float64 {
	n := len(a)

	if n == 0 {
		return 0
	}

	limit := alignedLen(n)
	result := 0.0

	if useAVX2 {
		if limit > 0 {
			result = dotAVX2(a[:limit], b[:limit])
		}
	} else if limit > 0 {
		result = dotSSE2(a[:limit], b[:limit])
	}

	for idx := limit; idx < n; idx++ {
		result += a[idx] * b[idx]
	}

	return result
}

/*
applySubVec computes dst = a - b elementwise.
*/
func applySubVec(dst, a, b []float64) {
	limit := alignedLen(len(a))

	if useAVX2 {
		if limit > 0 {
			subVecAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		subVecSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for idx := limit; idx < len(a); idx++ {
		dst[idx] = a[idx] - b[idx]
	}
}

/*
alignedLen returns the largest multiple of the SIMD width <= n.
*/
func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}

/*
applyMatMulFull computes C = A @ B where A is [m x k], B is [k x n], C is [m x n].
All matrices are row-major flat slices.
*/
func applyMatMulFull(dst, a, b []float64, m, k, n int) {
	for row := 0; row < m; row++ {
		dstRow := dst[row*n : (row+1)*n]
		aRow := a[row*k : (row+1)*k]

		for col := 0; col < n; col++ {
			dstRow[col] = 0
		}

		for kIdx := 0; kIdx < k; kIdx++ {
			scale := aRow[kIdx]
			bRow := b[kIdx*n : (kIdx+1)*n]
			applyAxpy(dstRow, bRow, scale)
		}
	}
}

/*
applyMatMulTransposeLeft computes C = A^T @ B where A is [t x p], B is [t x q], C is [p x q].
Equivalent to computing the cross product A^T B.
*/
func applyMatMulTransposeLeft(dst, a, b []float64, t, p, q int) {
	for row := 0; row < p; row++ {
		for col := 0; col < q; col++ {
			dst[row*q+col] = 0
		}
	}

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		aRow := a[obsIdx*p : (obsIdx+1)*p]
		bRow := b[obsIdx*q : (obsIdx+1)*q]

		for pIdx := 0; pIdx < p; pIdx++ {
			applyAxpy(dst[pIdx*q:(pIdx+1)*q], bRow, aRow[pIdx])
		}
	}
}

/*
applyMatVecTranspose computes dst = A^T @ x where A is [t x p] and x is [t].
*/
func applyMatVecTranspose(dst, a, x []float64, t, p int) {
	for pIdx := 0; pIdx < p; pIdx++ {
		dst[pIdx] = 0
	}

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		applyAxpy(dst, a[obsIdx*p:(obsIdx+1)*p], x[obsIdx])
	}
}
