//go:build arm64

package neon

//go:noescape
func AbsFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func NegFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func SqrtFloat32NEONAsm(dst, src *float32, n int)

//go:noescape
func ReluFloat32NEONAsm(dst, src *float32, n int)
