//go:build amd64

package predictive_coding

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func matVecAVX2(dst, W, x []float64, rows, cols int)

//go:noescape
func matVecSSE2(dst, W, x []float64, rows, cols int)

//go:noescape
func matVecTransposeAVX2(dst, W, x []float64, rows, cols int)

//go:noescape
func matVecTransposeSSE2(dst, W, x []float64, rows, cols int)

//go:noescape
func subVecAVX2(dst, a, b []float64)

//go:noescape
func subVecSSE2(dst, a, b []float64)

//go:noescape
func mulVecAVX2(dst, a, b []float64)

//go:noescape
func mulVecSSE2(dst, a, b []float64)

//go:noescape
func axpyAVX2(dst, src []float64, scale float64)

//go:noescape
func axpySSE2(dst, src []float64, scale float64)

//go:noescape
func outerRowAVX2(dst, b []float64, scale float64)

//go:noescape
func outerRowSSE2(dst, b []float64, scale float64)

func alignedLen(n int) int {
	width := 2
	if useAVX2 {
		width = 4
	}
	return n / width * width
}

func applyMatVec(dst, W, x []float64, rows, cols int) {
	if useAVX2 && useFMA {
		matVecAVX2(dst, W, x, rows, cols)
	} else {
		matVecSSE2(dst, W, x, rows, cols)
	}
}

func applyMatVecTranspose(dst, W, x []float64, rows, cols int) {
	if useAVX2 && useFMA {
		matVecTransposeAVX2(dst, W, x, rows, cols)
	} else {
		matVecTransposeSSE2(dst, W, x, rows, cols)
	}
}

func applySubVec(dst, a, b []float64) {
	n := len(a)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			subVecAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		subVecSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for i := limit; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

func applySubVecInPlace(dst, src []float64) {
	applySubVec(dst, dst, src)
}

func applyMulVec(dst, a, b []float64) {
	n := len(a)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			mulVecAVX2(dst[:limit], a[:limit], b[:limit])
		}
	} else if limit > 0 {
		mulVecSSE2(dst[:limit], a[:limit], b[:limit])
	}

	for i := limit; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func applyAxpy(dst, src []float64, scale float64) {
	n := len(src)
	limit := alignedLen(n)

	if useAVX2 {
		if limit > 0 {
			axpyAVX2(dst[:limit], src[:limit], scale)
		}
	} else if limit > 0 {
		axpySSE2(dst[:limit], src[:limit], scale)
	}

	for i := limit; i < n; i++ {
		dst[i] += scale * src[i]
	}
}

func applyOuterAdd(W, eps, r []float64, lr float64, dOut, dIn int) {
	for i := 0; i < dOut; i++ {
		scale := lr * eps[i]
		row := W[i*dIn : (i+1)*dIn]
		limit := alignedLen(dIn)

		if useAVX2 {
			if limit > 0 {
				outerRowAVX2(row[:limit], r[:limit], scale)
			}
		} else if limit > 0 {
			outerRowSSE2(row[:limit], r[:limit], scale)
		}

		for j := limit; j < dIn; j++ {
			row[j] += scale * r[j]
		}
	}
}
