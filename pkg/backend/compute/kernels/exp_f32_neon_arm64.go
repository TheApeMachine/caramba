//go:build arm64

package kernels

//go:noescape
func expFloat32NEONAsm(dst, src *float32, n int)
