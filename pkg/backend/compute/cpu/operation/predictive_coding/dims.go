package predictive_coding

import "fmt"

func rowMajorWeightLen(dOut, dIn int) int {
	if dOut <= 0 || dIn <= 0 {
		panic(fmt.Sprintf(
			"predictive_coding: D_out and D_in must be positive, got D_out=%d D_in=%d",
			dOut, dIn,
		))
	}

	prod := int64(dOut) * int64(dIn)
	maxInt := int64(^uint(0) >> 1)

	if prod > maxInt {
		panic(fmt.Sprintf(
			"predictive_coding: D_out*D_in overflows int (D_out=%d D_in=%d)",
			dOut, dIn,
		))
	}

	return int(prod)
}

func requireMatVec(dst, w, x []float64, rows, cols int) {
	needW := rowMajorWeightLen(rows, cols)

	if len(dst) < rows || len(w) < needW || len(x) < cols {
		panic(fmt.Sprintf(
			"predictive_coding: applyMatVec: need len(dst)>=%d len(W)>=%d len(x)>=%d; got %d %d %d",
			rows, needW, cols, len(dst), len(w), len(x),
		))
	}
}

func requireMatVecTranspose(dst, w, x []float64, rows, cols int) {
	needW := rowMajorWeightLen(rows, cols)

	if len(dst) < cols || len(w) < needW || len(x) < rows {
		panic(fmt.Sprintf(
			"predictive_coding: applyMatVecTranspose: need len(dst)>=%d len(W)>=%d len(x)>=%d; got %d %d %d",
			cols, needW, rows, len(dst), len(w), len(x),
		))
	}
}

func requireEqualLen3(dst, a, b []float64, name string) {
	n := len(a)

	if len(dst) != n || len(b) != n {
		panic(fmt.Sprintf(
			"predictive_coding: %s: need equal lengths, got len(dst)=%d len(a)=%d len(b)=%d",
			name, len(dst), len(a), len(b),
		))
	}
}

func requireAxpy(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf(
			"predictive_coding: applyAxpy: need len(dst)>=%d, got len(dst)=%d len(src)=%d",
			len(src), len(dst), len(src),
		))
	}
}

func requireOuterAdd(w, eps, r []float64, dOut, dIn int) {
	needW := rowMajorWeightLen(dOut, dIn)

	if len(w) < needW || len(eps) < dOut || len(r) < dIn {
		panic(fmt.Sprintf(
			"predictive_coding: applyOuterAdd: need len(W)>=%d len(eps)>=%d len(r)>=%d; got %d %d %d",
			needW, dOut, dIn, len(w), len(eps), len(r),
		))
	}
}
