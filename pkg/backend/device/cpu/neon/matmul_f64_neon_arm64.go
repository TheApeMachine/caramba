//go:build arm64

package neon

//go:noescape
func MatmulRowFloat64NEONAsm(cRow, aRow, b *float64, inner, cols int)
