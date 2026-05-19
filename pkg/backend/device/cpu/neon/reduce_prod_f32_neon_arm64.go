//go:build arm64

package neon

//go:noescape
func ReduceProdFloat32NEONAsm(src *float32, n int) float32
