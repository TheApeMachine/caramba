//go:build arm64

package neon

//go:noescape
func SigmoidFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func SiluFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func TanhFloat32NEONAsm(dst, src *float32, n int)
