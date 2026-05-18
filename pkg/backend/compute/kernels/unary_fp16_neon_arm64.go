//go:build arm64

package kernels

//go:noescape
func absFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func negFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func sqrtFloat16NEONAsm(dst, src *uint16, n int)

//go:noescape
func reluFloat16NEONAsm(dst, src *uint16, n int)
