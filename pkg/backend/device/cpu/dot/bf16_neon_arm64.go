//go:build arm64

package dot

//go:noescape
func DotBFloat16NEONAsm(a, b *uint16, n int) uint16
