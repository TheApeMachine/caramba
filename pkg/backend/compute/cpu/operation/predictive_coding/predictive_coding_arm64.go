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

//go:noescape
func outerRowNEON(dst, b []float64, scale float64)

func applyMatVec(dst, W, x []float64, rows, cols int) {
	matVecNEON(dst, W, x, rows, cols)
}

func applyMatVecTranspose(dst, W, x []float64, rows, cols int) {
	matVecTransposeNEON(dst, W, x, rows, cols)
}

func applySubVec(dst, a, b []float64) {
	subVecNEON(dst, a, b)
}

func applySubVecInPlace(dst, src []float64) {
	subVecNEON(dst, dst, src)
}

func applyMulVec(dst, a, b []float64) {
	mulVecNEON(dst, a, b)
}

func applyAxpy(dst, src []float64, scale float64) {
	axpyNEON(dst, src, scale)
}

func applyOuterAdd(W, eps, r []float64, lr float64, dOut, dIn int) {
	for i := 0; i < dOut; i++ {
		scale := lr * eps[i]
		row := W[i*dIn : (i+1)*dIn]
		outerRowNEON(row, r, scale)
	}
}
