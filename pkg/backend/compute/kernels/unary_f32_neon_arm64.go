//go:build arm64

package kernels

//go:noescape
func absFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func negFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func sqrtFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func reluFloat32NEONAsm(dst, src *float32, n int)
