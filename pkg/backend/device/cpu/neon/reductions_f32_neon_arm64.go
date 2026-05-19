//go:build arm64

package neon

//go:noescape
func SumFloat32NEONAsm(src *float32, n int) float32
