//go:build arm64

package kernels

//go:noescape
func adamStepFloat32NEONAsm(
	params, grad, first, second, output *float32,
	n int,
	lr, beta1, beta2, eps, beta1Corr, beta2Corr float32,
)

//go:noescape
func sgdStepFloat32NEONAsm(
	params, grad, momentum, output *float32,
	n int,
	lr, momentumFactor, weightDecay float32,
)

//go:noescape
func adamwStepFloat32NEONAsm(
	params, grad, first, second, output *float32,
	n int,
	lr, beta1, beta2, eps, beta1Corr, beta2Corr, weightDecay float32,
)
