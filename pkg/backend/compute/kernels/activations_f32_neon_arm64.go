//go:build arm64

package kernels

//go:noescape
func sigmoidFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func siluFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func tanhFloat32NEONAsm(dst, src *float32, n int)
