//go:build arm64

package neon

//go:noescape
func DotInt8NEONAsm(a, b *int8, n int) int32
