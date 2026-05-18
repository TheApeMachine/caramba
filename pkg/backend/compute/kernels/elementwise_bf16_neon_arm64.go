//go:build arm64

package kernels

//go:noescape
func addBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func subBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func mulBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func divBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func maxBFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func minBFloat16NEONAsm(dst, left, right *uint16, n int)
