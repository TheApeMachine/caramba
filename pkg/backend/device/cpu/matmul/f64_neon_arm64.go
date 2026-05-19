//go:build arm64

package matmul

//go:noescape
func MatmulRowFloat64NEONAsm(cRow, aRow, b *float64, inner, cols int)
