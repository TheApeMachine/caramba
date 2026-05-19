//go:build amd64

package optimizer

//go:noescape
func AdamStepFloat32AVX512Asm(
	params, grad, first, second, output *float32,
	count int,
	learningRate, beta1, beta2, epsilon, beta1Correction, beta2Correction float32,
)

//go:noescape
func SgdStepFloat32AVX512Asm(
	params, grad, momentum, output *float32,
	count int,
	learningRate, momentumFactor, weightDecay float32,
)

//go:noescape
func AdamwStepFloat32AVX512Asm(
	params, grad, first, second, output *float32,
	count int,
	learningRate, beta1, beta2, epsilon, beta1Correction, beta2Correction, weightDecay float32,
)

//go:noescape
func AdamaxStepFloat32AVX512Asm(
	params, grad, first, infinity, output *float32,
	count int,
	learningRate, beta1, beta2, epsilon, beta1Correction float32,
)

//go:noescape
func AdagradStepFloat32AVX512Asm(
	params, grad, accum, output *float32,
	count int,
	learningRate, epsilon float32,
)

//go:noescape
func RmspropStepFloat32AVX512Asm(
	params, grad, second, output *float32,
	count int,
	learningRate, decay, epsilon float32,
)

//go:noescape
func LionStepFloat32AVX512Asm(
	params, grad, momentum, output *float32,
	count int,
	learningRate, beta1, beta2, weightDecay float32,
)

//go:noescape
func LbfgsStepFloat32AVX512Asm(
	params, grad, output *float32,
	count int,
	learningRate float32,
)

//go:noescape
func LarsStepFloat32AVX512Asm(
	params, grad, momentum, output *float32,
	count int,
	learningRate, momentumFactor, weightDecay, effectiveLearningRate float32,
)

//go:noescape
func HebbianStepRowFloat32AVX512Asm(
	weights, pre, output *float32,
	count int,
	decayFactor, lrPost float32,
)
