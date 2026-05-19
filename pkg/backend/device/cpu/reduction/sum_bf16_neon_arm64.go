//go:build arm64

package reduction

//go:noescape
func SumBFloat16NEONAsm(src *uint16, n int) uint16
