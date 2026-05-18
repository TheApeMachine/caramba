//go:build arm64

package kernels

//go:noescape
func matmulRowFloat64NEONAsm(cRow, aRow, b *float64, inner, cols int)
