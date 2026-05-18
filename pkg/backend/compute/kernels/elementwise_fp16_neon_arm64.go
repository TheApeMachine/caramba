//go:build arm64

package kernels

//go:noescape
func addFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func subFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func mulFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func divFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func maxFloat16NEONAsm(dst, left, right *uint16, n int)

//go:noescape
func minFloat16NEONAsm(dst, left, right *uint16, n int)
