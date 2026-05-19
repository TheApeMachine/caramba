//go:build !arm64

package cpu

func MatmulFloat32Native(out, left, right []float32, rows, inner, cols int) {
	for row := 0; row < rows; row++ {
		for k := 0; k < inner; k++ {
			leftValue := left[row*inner+k]
			for col := 0; col < cols; col++ {
				out[row*cols+col] += leftValue * right[k*cols+col]
			}
		}
	}
}
