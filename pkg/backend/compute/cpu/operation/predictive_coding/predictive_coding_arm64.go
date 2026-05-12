//go:build arm64

package predictive_coding

//go:noescape
func matVecNEON(dst, W, x []float64, rows, cols int)

//go:noescape
func matVecTransposeNEON(dst, W, x []float64, rows, cols int)

//go:noescape
func subVecNEON(dst, a, b []float64)

//go:noescape
func mulVecNEON(dst, a, b []float64)

//go:noescape
func axpyNEON(dst, src []float64, scale float64)

// applyMatVec sets dst[i] = dot(W[i,:],x) for i in [0,rows). W is row-major with rows×cols
// elements. Requires len(dst)>=rows, len(W)>=rows*cols, len(x)>=cols. Overwrites dst.
func applyMatVec(dst, W, x []float64, rows, cols int) {
	requireMatVec(dst, W, x, rows, cols)
	matVecNEON(dst, W, x, rows, cols)
}

// applyMatVecTranspose sets dst[j] = sum_i W[i,j]*x[i] (W^T @ x). W is row-major rows×cols;
// dst has length cols. Requires len(dst)>=cols, len(W)>=rows*cols, len(x)>=rows. Zeros dst
// then accumulates (assembly).
func applyMatVecTranspose(dst, W, x []float64, rows, cols int) {
	requireMatVecTranspose(dst, W, x, rows, cols)
	matVecTransposeNEON(dst, W, x, rows, cols)
}

// applySubVec sets dst[k]=a[k]-b[k] for all k. Requires len(dst)==len(a)==len(b). Overwrites dst.
func applySubVec(dst, a, b []float64) {
	requireEqualLen3(dst, a, b, "applySubVec")
	subVecNEON(dst, a, b)
}

// applySubVecInPlace sets dst[k] -= src[k]. Requires len(dst)==len(src). Overwrites dst.
func applySubVecInPlace(dst, src []float64) {
	requireEqualLen3(dst, dst, src, "applySubVecInPlace")
	subVecNEON(dst, dst, src)
}

// applyMulVec sets dst[k]=a[k]*b[k]. Requires len(dst)==len(a)==len(b). Overwrites dst.
func applyMulVec(dst, a, b []float64) {
	requireEqualLen3(dst, a, b, "applyMulVec")
	mulVecNEON(dst, a, b)
}

// applyAxpy performs dst[i] += scale*src[i]. Requires len(dst)>=len(src). Overwrites dst.
func applyAxpy(dst, src []float64, scale float64) {
	requireAxpy(dst, src)
	axpyNEON(dst, src, scale)
}

// applyOuterAdd updates W in place: for each row i, W[i*dIn+j] += lr*eps[i]*r[j].
// Requires len(W)>=dOut*dIn, len(eps)>=dOut, len(r)>=dIn. Reuses axpyNEON per row (same op as dst+=scale*src).
func applyOuterAdd(W, eps, r []float64, lr float64, dOut, dIn int) {
	requireOuterAdd(W, eps, r, dOut, dIn)

	for i := 0; i < dOut; i++ {
		scale := lr * eps[i]
		row := W[i*dIn : (i+1)*dIn]
		axpyNEON(row, r, scale)
	}
}
