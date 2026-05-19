//go:build arm64

package neon

//go:noescape
func Laplacian1DStencilF32NEONAsm(out, left, center, right *float32, invH2 float32, n int)

//go:noescape
func Grad1DStencilF32NEONAsm(out, left, right *float32, invTwoDx float32, n int)

//go:noescape
func Laplacian4StencilF32NEONAsm(out, um2, um1, u0, up1, up2 *float32, invDen float32, n int)
