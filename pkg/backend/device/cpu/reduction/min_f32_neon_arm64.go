//go:build arm64

package reduction

//go:noescape
func ReduceMinFloat32NEONAsm(src *float32, n int) float32
