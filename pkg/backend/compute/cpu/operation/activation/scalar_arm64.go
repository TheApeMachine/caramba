//go:build arm64

package activation

//go:noescape
func scalarTanhARM64(dst, src []float64)

//go:noescape
func scalarSigmoidARM64(dst, src []float64)

//go:noescape
func scalarReLUARM64(dst, src []float64)

//go:noescape
func scalarLeakyReLUARM64(dst, src []float64, alpha float64)

//go:noescape
func scalarGeLUARM64(dst, src []float64)

//go:noescape
func scalarSwiGLUARM64(dst, src []float64)

func scalarTanhKernel(dst, src []float64)    { scalarTanhARM64(dst, src) }
func scalarSigmoidKernel(dst, src []float64) { scalarSigmoidARM64(dst, src) }
func scalarReLUKernel(dst, src []float64)    { scalarReLUARM64(dst, src) }
func scalarLeakyReLUKernel(dst, src []float64, alpha float64) {
	scalarLeakyReLUARM64(dst, src, alpha)
}
func scalarGeLUKernel(dst, src []float64)   { scalarGeLUARM64(dst, src) }
func scalarSwiGLUKernel(dst, src []float64) { scalarSwiGLUARM64(dst, src) }
