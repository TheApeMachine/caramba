//go:build arm64

package elementwise

//go:noescape
func AxpyFloat32NEONAsm(y, x *float32, alpha float32, n int)
