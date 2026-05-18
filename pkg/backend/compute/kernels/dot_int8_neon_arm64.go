//go:build arm64

package kernels

//go:noescape
func dotInt8NEONAsm(a, b *int8, n int) int32
