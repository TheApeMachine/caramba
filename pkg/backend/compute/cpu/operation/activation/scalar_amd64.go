//go:build amd64

package activation

//go:noescape
func scalarTanhAMD64(dst, src []float64)

//go:noescape
func scalarSigmoidAMD64(dst, src []float64)

//go:noescape
func scalarReLUAMD64(dst, src []float64)

//go:noescape
func scalarLeakyReLUAMD64(dst, src []float64, alpha float64)

//go:noescape
func scalarGeLUAMD64(dst, src []float64)

//go:noescape
func scalarSwiGLUAMD64(dst, src []float64)

func scalarTanhKernel(dst, src []float64)    { scalarTanhAMD64(dst, src) }
func scalarSigmoidKernel(dst, src []float64) { scalarSigmoidAMD64(dst, src) }
func scalarReLUKernel(dst, src []float64)    { scalarReLUAMD64(dst, src) }
func scalarLeakyReLUKernel(dst, src []float64, alpha float64) {
	scalarLeakyReLUAMD64(dst, src, alpha)
}
func scalarGeLUKernel(dst, src []float64)   { scalarGeLUAMD64(dst, src) }
func scalarSwiGLUKernel(dst, src []float64) { scalarSwiGLUAMD64(dst, src) }
