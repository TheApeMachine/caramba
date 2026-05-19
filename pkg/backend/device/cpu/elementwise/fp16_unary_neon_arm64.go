//go:build arm64

package elementwise

//go:noescape
func AbsFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func NegFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func SqrtFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func ReluFloat16NEONAsm(dst, src *uint16, n int)
