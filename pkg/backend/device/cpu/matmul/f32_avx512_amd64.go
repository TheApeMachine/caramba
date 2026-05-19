//go:build amd64

package matmul

//go:noescape
func MatmulRowFloat32AVX512Asm(cRow, aRow, b *float32, inner, cols int)

/*
MatmulFloat32AVX512 computes out += left × right with AVX-512 row kernels.
*/
func MatmulFloat32AVX512(out, left, right []float32, rows, inner, cols int) {
	if rows == 0 || inner == 0 || cols == 0 {
		return
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		MatmulRowFloat32AVX512Asm(
			&out[rowIndex*cols],
			&left[rowIndex*inner],
			&right[0],
			inner,
			cols,
		)
	}
}
