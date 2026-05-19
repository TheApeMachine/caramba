//go:build arm64

package elementwise

//go:noescape
func AddFloat64NEONAsm(dst, left, right *float64, n int)
