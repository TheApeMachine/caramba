//go:build arm64

package reduction

//go:noescape
func L1NormNEONAsm(src *float32, count int) float32
