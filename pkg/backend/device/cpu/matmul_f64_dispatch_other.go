//go:build !arm64

package cpu

func MatmulFloat64Native(out, left, right []float64, rows, inner, cols int) {
	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] +=
					leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}
