//go:build !amd64 && !arm64

package math

func matmulKernel(dst, left, right []float64, rows, inner, cols int) {
	for row := range rows {
		for col := range cols {
			sum := 0.0

			for index := range inner {
				sum += left[row*inner+index] * right[index*cols+col]
			}

			dst[row*cols+col] = sum
		}
	}
}
