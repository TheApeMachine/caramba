//go:build arm64

package kernels

//go:noescape
func matmulRowFloat32NEONAsm(cRow, aRow, b *float32, inner, cols int)
