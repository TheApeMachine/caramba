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

//go:noescape
func adamaxStepFloat32NEONAsm(
	params, grad, first, infinity, output *float32,
	n int,
	lr, beta1, beta2, eps, beta1Corr float32,
)

//go:noescape
func adagradStepFloat32NEONAsm(
	params, grad, accum, output *float32,
	n int,
	lr, eps float32,
)

//go:noescape
func rmspropStepFloat32NEONAsm(
	params, grad, second, output *float32,
	n int,
	lr, decay, eps float32,
)

//go:noescape
func lionStepFloat32NEONAsm(
	params, grad, momentum, output *float32,
	n int,
	lr, beta1, beta2, weightDecay float32,
)

//go:noescape
func lbfgsStepFloat32NEONAsm(
	params, grad, output *float32,
	n int,
	lr float32,
)

//go:noescape
func larsStepFloat32NEONAsm(
	params, grad, momentum, output *float32,
	n int,
	lr, momentumFactor, weightDecay, effectiveLr float32,
)

//go:noescape
func hebbianStepRowFloat32NEONAsm(
	weights, pre, output *float32,
	n int,
	decayFactor, lrPost float32,
)
