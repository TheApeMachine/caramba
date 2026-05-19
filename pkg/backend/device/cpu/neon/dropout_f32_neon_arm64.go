//go:build arm64

package neon

//go:noescape
func DropoutFloat32NEONAsm(dst, src *float32, n int, seedState *uint32, scale, threshold float32)
