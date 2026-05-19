//go:build arm64

package matmul

//go:noescape
func MatmulRowFloat32NEONAsm(cRow, aRow, b *float32, inner, cols int)
