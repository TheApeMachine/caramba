//go:build arm64

package neon

//go:noescape
func AddFloat64NEONAsm(dst, left, right *float64, n int)
