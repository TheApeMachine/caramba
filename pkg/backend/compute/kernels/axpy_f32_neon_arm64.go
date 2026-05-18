//go:build arm64

package kernels

//go:noescape
func axpyFloat32NEONAsm(y, x *float32, alpha float32, n int)
