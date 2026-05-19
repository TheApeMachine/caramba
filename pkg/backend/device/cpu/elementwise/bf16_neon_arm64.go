//go:build arm64

package elementwise

//go:noescape
func AddBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func SubBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func MulBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func DivBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func MaxBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func MinBFloat16NEONAsm(dst, left, right *uint16, n int)
