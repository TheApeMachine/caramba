//go:build arm64

package activation

//go:noescape
func scalarTanhNEON(dst, src []float64)

//go:noescape
func scalarSigmoidNEON(dst, src []float64)

//go:noescape
func scalarReLUNEON(dst, src []float64)

//go:noescape
func scalarLeakyReLUNEON(dst, src []float64, alpha float64)

//go:noescape
func scalarGeLUNEON(dst, src []float64)

//go:noescape
func scalarSwiGLUNEON(dst, src []float64)

func scalarTanhKernel(dst, src []float64)    { scalarTanhNEON(dst, src) }
func scalarSigmoidKernel(dst, src []float64) { scalarSigmoidNEON(dst, src) }
func scalarReLUKernel(dst, src []float64)    { scalarReLUNEON(dst, src) }
func scalarLeakyReLUKernel(dst, src []float64, alpha float64) {
	scalarLeakyReLUNEON(dst, src, alpha)
}
func scalarGeLUKernel(dst, src []float64)   { scalarGeLUNEON(dst, src) }
func scalarSwiGLUKernel(dst, src []float64) { scalarSwiGLUNEON(dst, src) }
