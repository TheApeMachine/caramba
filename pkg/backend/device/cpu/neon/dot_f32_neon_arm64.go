//go:build arm64

package neon

//go:noescape
func DotFloat32NEONAsm(a, b *float32, n int) float32
