//go:build arm64

package neon

//go:noescape
func AdamStepFloat32NEONAsm(
	params, grad, first, second, output *float32,
	n int,
	lr, beta1, beta2, eps, beta1Corr, beta2Corr float32,
)

//go:noescape
func SgdStepFloat32NEONAsm(
	params, grad, momentum, output *float32,
	n int,
	lr, momentumFactor, weightDecay float32,
)

//go:noescape
func AdamwStepFloat32NEONAsm(
	params, grad, first, second, output *float32,
	n int,
	lr, beta1, beta2, eps, beta1Corr, beta2Corr, weightDecay float32,
)

//go:noescape
func AdamaxStepFloat32NEONAsm(
	params, grad, first, infinity, output *float32,
	n int,
	lr, beta1, beta2, eps, beta1Corr float32,
)

//go:noescape
func AdagradStepFloat32NEONAsm(
	params, grad, accum, output *float32,
	n int,
	lr, eps float32,
)

//go:noescape
func RmspropStepFloat32NEONAsm(
	params, grad, second, output *float32,
	n int,
	lr, decay, eps float32,
)

//go:noescape
func LionStepFloat32NEONAsm(
	params, grad, momentum, output *float32,
	n int,
	lr, beta1, beta2, weightDecay float32,
)

//go:noescape
func LbfgsStepFloat32NEONAsm(
	params, grad, output *float32,
	n int,
	lr float32,
)

//go:noescape
func LarsStepFloat32NEONAsm(
	params, grad, momentum, output *float32,
	n int,
	lr, momentumFactor, weightDecay, effectiveLr float32,
)

//go:noescape
func HebbianStepRowFloat32NEONAsm(
	weights, pre, output *float32,
	n int,
	decayFactor, lrPost float32,
)
