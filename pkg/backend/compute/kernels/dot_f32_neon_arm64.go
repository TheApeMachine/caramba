//go:build arm64

package kernels

//go:noescape
func dotFloat32NEONAsm(a, b *float32, n int) float32
