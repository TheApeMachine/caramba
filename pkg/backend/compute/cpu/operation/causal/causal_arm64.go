//go:build arm64

package causal

//go:noescape
func matVecNEON(dst, w, x []float64, rows, cols int)

//go:noescape
func axpyNEON(dst, src []float64, scale float64)

//go:noescape
func dotNEON(a, b []float64) float64

//go:noescape
func subVecNEON(dst, a, b []float64)

/*
applyMatVec computes dst = W @ x where W is [rows x cols] row-major and x is [cols].
*/
func applyMatVec(dst, w, x []float64, rows, cols int) {
	matVecNEON(dst, w, x, rows, cols)
}

/*
applyAxpy computes dst += scale * src elementwise.
*/
func applyAxpy(dst, src []float64, scale float64) {
	if len(dst) < len(src) {
		panic("causal: applyAxpy: len(dst) < len(src)")
	}

	n := len(src)
	limit := n / 2 * 2

	if limit > 0 {
		axpyNEON(dst[:limit], src[:limit], scale)
	}

	if n%2 != 0 {
		dst[n-1] += scale * src[n-1]
	}
}

/*
applyDotProduct computes the inner product of a and b.
*/
func applyDotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("causal: applyDotProduct: length mismatch")
	}

	n := len(a)

	if n == 0 {
		return 0
	}

	limit := n / 2 * 2
	result := 0.0

	if limit > 0 {
		result = dotNEON(a[:limit], b[:limit])
	}

	if n%2 != 0 {
		result += a[n-1] * b[n-1]
	}

	return result
}

/*
applySubVec computes dst = a - b elementwise.
*/
func applySubVec(dst, a, b []float64) {
	if len(dst) < len(a) || len(b) < len(a) {
		panic("causal: applySubVec: slice too short")
	}

	n := len(a)
	limit := n / 2 * 2

	if limit > 0 {
		subVecNEON(dst[:limit], a[:limit], b[:limit])
	}

	if n%2 != 0 {
		dst[n-1] = a[n-1] - b[n-1]
	}
}

/*
applyMatMulFull computes C = A @ B where A is [m x k], B is [k x n], C is [m x n].
*/
func applyMatMulFull(dst, a, b []float64, m, k, n int) {
	if m < 0 || k < 0 || n < 0 {
		panic("causal: applyMatMulFull: negative dimension")
	}

	needDst := m * n
	needA := m * k
	needB := k * n

	if len(dst) < needDst || len(a) < needA || len(b) < needB {
		panic("causal: applyMatMulFull: slice shorter than declared matrix size")
	}

	for row := 0; row < m; row++ {
		dstRow := dst[row*n : (row+1)*n]
		aRow := a[row*k : (row+1)*k]

		for i := range dstRow {
			dstRow[i] = 0
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
*/
func applyMatMulTransposeLeft(dst, a, b []float64, t, p, q int) {
	if p < 0 || q < 0 || t < 0 {
		panic("causal: applyMatMulTransposeLeft: negative dimension")
	}

	needDst := p * q
	needA := t * p
	needB := t * q

	if len(dst) < needDst || len(a) < needA || len(b) < needB {
		panic("causal: applyMatMulTransposeLeft: slice shorter than declared matrix size")
	}

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
	if t < 0 || p < 0 {
		panic("causal: applyMatVecTranspose: negative dimension")
	}

	if len(dst) < p || len(a) < t*p || len(x) < t {
		panic("causal: applyMatVecTranspose: slice shorter than declared size")
	}

	for pIdx := 0; pIdx < p; pIdx++ {
		dst[pIdx] = 0
	}

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		applyAxpy(dst, a[obsIdx*p:(obsIdx+1)*p], x[obsIdx])
	}
}
