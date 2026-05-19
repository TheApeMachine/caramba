//go:build arm64

package reduction

//go:noescape
func SumFloat32NEONAsm(src *float32, n int) float32
