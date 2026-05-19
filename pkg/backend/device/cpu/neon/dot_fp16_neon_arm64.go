//go:build arm64

package neon

//go:noescape
func DotFloat16NEONAsm(a, b *uint16, n int) uint16
