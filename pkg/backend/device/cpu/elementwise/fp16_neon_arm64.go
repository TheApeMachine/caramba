//go:build arm64

package elementwise

//go:noescape
func AddFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func SubFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func MulFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func DivFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func MaxFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func MinFloat16NEONAsm(dst, left, right *uint16, n int)
