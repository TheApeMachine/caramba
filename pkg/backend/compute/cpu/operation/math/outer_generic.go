//go:build !amd64 && !arm64

package math

func outerKernel(out, left, right []float64, rows, cols int) {
	for row := range rows {
		for col := range cols {
			out[row*cols+col] = left[row] * right[col]
		}
	}
}
