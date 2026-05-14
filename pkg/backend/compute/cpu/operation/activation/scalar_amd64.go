//go:build amd64

package activation

//go:noescape
func scalarTanhAVX2(dst, src []float64)

//go:noescape
func scalarSigmoidAVX2(dst, src []float64)

//go:noescape
func scalarReLUAVX2(dst, src []float64)

//go:noescape
func scalarLeakyReLUAVX2(dst, src []float64, alpha float64)

//go:noescape
func scalarGeLUAVX2(dst, src []float64)

//go:noescape
func scalarSwiGLUAVX2(dst, src []float64)

func scalarTanhKernel(dst, src []float64)    { scalarTanhAVX2(dst, src) }
func scalarSigmoidKernel(dst, src []float64) { scalarSigmoidAVX2(dst, src) }
func scalarReLUKernel(dst, src []float64)    { scalarReLUAVX2(dst, src) }
func scalarLeakyReLUKernel(dst, src []float64, alpha float64) {
	scalarLeakyReLUAVX2(dst, src, alpha)
}
func scalarGeLUKernel(dst, src []float64)   { scalarGeLUAVX2(dst, src) }
func scalarSwiGLUKernel(dst, src []float64) { scalarSwiGLUAVX2(dst, src) }
