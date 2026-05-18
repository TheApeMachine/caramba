//go:build arm64

package kernels

//go:noescape
func absBFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func negBFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func sqrtBFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func reluBFloat16NEONAsm(dst, src *uint16, n int)
