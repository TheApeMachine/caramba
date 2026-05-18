//go:build arm64

package kernels

//go:noescape
func dotBFloat16NEONAsm(a, b *uint16, n int) uint16
