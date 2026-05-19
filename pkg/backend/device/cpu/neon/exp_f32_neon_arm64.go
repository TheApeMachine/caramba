//go:build arm64

package neon

//go:noescape
func ExpFloat32NEONAsm(dst, src *float32, n int)
