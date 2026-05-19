//go:build arm64

package neon

//go:noescape
func ReduceMinFloat32NEONAsm(src *float32, n int) float32
