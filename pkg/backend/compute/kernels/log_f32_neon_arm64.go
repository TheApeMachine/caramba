//go:build arm64

package kernels

//go:noescape
func logFloat32NEONAsm(dst, src *float32, n int)
