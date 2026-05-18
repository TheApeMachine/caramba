//go:build arm64

package kernels

//go:noescape
func sumBFloat16NEONAsm(src *uint16, n int) uint16
