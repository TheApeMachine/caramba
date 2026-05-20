//go:build amd64

package physics

//go:noescape
func Laplacian1DStencilF32AVX512Asm(out, left, center, right *float32, invH2 float32, n int)

//go:noescape
func Grad1DStencilF32AVX512Asm(out, left, right *float32, invTwoDx float32, n int)

//go:noescape
func Laplacian4StencilF32AVX512Asm(out, um2, um1, u0, up1, up2 *float32, invDen float32, n int)
