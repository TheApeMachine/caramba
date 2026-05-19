//go:build arm64

package neon

//go:noescape
func AxpyFloat32NEONAsm(y, x *float32, alpha float32, n int)
