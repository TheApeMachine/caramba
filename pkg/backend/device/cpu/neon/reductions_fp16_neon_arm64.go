//go:build arm64

package neon

//go:noescape
func SumFloat16NEONAsm(src *uint16, n int) uint16
