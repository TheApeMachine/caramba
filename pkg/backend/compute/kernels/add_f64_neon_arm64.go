//go:build arm64

package kernels

//go:noescape
func addFloat64NEONAsm(dst, left, right *float64, n int)
