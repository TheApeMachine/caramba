//go:build arm64

package elementwise

//go:noescape
func AbsBFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func NegBFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func SqrtBFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func ReluBFloat16NEONAsm(dst, src *uint16, n int)
