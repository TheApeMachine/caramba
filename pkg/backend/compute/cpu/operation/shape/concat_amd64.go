//go:build amd64

package shape

func concatKernel(dst []float64, inputs [][]float64, outer, dimSize, inner int) {
	concatGenericKernel(dst, inputs, outer, dimSize, inner)
}
