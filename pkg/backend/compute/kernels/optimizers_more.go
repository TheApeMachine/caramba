package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Remaining optimizer kernels: Adamax, Adagrad, Adadelta, RMSprop,
NAdam, RAdam. Each registers under its canonical op name and
operates in fp32 master precision.

The configurations carry the scalar parameters; the default values
match standard transformer-training recipes for each optimizer.
*/

type AdamaxConfig struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	Step         int
}

func DefaultAdamaxConfig() AdamaxConfig {
	return AdamaxConfig{LearningRate: 2e-3, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, Step: 1}
}

type AdagradConfig struct {
	LearningRate float32
	Epsilon      float32
}

func DefaultAdagradConfig() AdagradConfig {
	return AdagradConfig{LearningRate: 1e-2, Epsilon: 1e-10}
}

type RMSpropConfig struct {
	LearningRate float32
	Decay        float32
	Epsilon      float32
}

func DefaultRMSpropConfig() RMSpropConfig {
	return RMSpropConfig{LearningRate: 1e-3, Decay: 0.99, Epsilon: 1e-8}
}

func init() {
	Default.Register(Kernel{
		Name: "adamax_step",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAdamaxDefault,
	})

	Default.Register(Kernel{
		Name: "adagrad_step",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAdagradDefault,
	})

	Default.Register(Kernel{
		Name: "rmsprop_step",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runRMSpropDefault,
	})
}

/*
AdamaxStepFloat32 applies one Adamax update (variant of Adam using
the infinity norm for the second moment).
*/
func AdamaxStepFloat32(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output tensor.Tensor,
) error {
	paramsView, gradView, firstView, infView, outView, err := adamViews(
		params, gradients, firstMoment, infinityMoment, output,
	)

	if err != nil {
		return err
	}

	adamaxStepSlices(config, paramsView, gradView, firstView, infView, outView)
	return nil
}

func adamaxStepSlicesScalar(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))

	for index, gradValue := range gradients {
		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue

		updated := config.Beta2 * infinityMoment[index]
		absGrad := float32(math.Abs(float64(gradValue)))

		if absGrad > updated {
			updated = absGrad
		}

		infinityMoment[index] = updated

		biasCorrectedFirst := firstMoment[index] / beta1Correction
		output[index] = params[index] - config.LearningRate*biasCorrectedFirst/(infinityMoment[index]+config.Epsilon)
	}
}

func runAdamaxDefault(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return AdamaxStepFloat32(
		DefaultAdamaxConfig(),
		args[0], args[1], args[2], args[3], args[4],
	)
}

/*
AdagradStepFloat32 accumulates squared gradients into a running state
and scales the step by the running root.
*/
func AdagradStepFloat32(
	config AdagradConfig,
	params, gradients, accumulator, output tensor.Tensor,
) error {
	paramsView, err := params.Float32Native()

	if err != nil {
		return err
	}

	gradView, err := gradients.Float32Native()

	if err != nil {
		return err
	}

	accView, err := accumulator.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(paramsView) != len(gradView) ||
		len(paramsView) != len(accView) ||
		len(paramsView) != len(outView) {
		return tensor.ErrShapeMismatch
	}

	adagradStepSlices(config, paramsView, gradView, accView, outView)
	return nil
}

func adagradStepSlicesScalar(
	config AdagradConfig,
	params, gradients, accumulator, output []float32,
) {
	for index, gradValue := range gradients {
		accumulator[index] += gradValue * gradValue
		denominator := float32(math.Sqrt(float64(accumulator[index]))) + config.Epsilon
		output[index] = params[index] - config.LearningRate*gradValue/denominator
	}
}

func runAdagradDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return AdagradStepFloat32(
		DefaultAdagradConfig(),
		args[0], args[1], args[2], args[3],
	)
}

/*
RMSpropStepFloat32 maintains an exponential moving average of the
squared gradient and scales the step by the average's root.
*/
func RMSpropStepFloat32(
	config RMSpropConfig,
	params, gradients, secondMoment, output tensor.Tensor,
) error {
	paramsView, err := params.Float32Native()

	if err != nil {
		return err
	}

	gradView, err := gradients.Float32Native()

	if err != nil {
		return err
	}

	secondView, err := secondMoment.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(paramsView) != len(gradView) ||
		len(paramsView) != len(secondView) ||
		len(paramsView) != len(outView) {
		return tensor.ErrShapeMismatch
	}

	rmspropStepSlices(config, paramsView, gradView, secondView, outView)
	return nil
}

func rmspropStepSlicesScalar(
	config RMSpropConfig,
	params, gradients, secondMoment, output []float32,
) {
	for index, gradValue := range gradients {
		gradSquared := gradValue * gradValue
		scaledGrad := (1 - config.Decay) * gradSquared
		secondMoment[index] = scaledGrad + config.Decay*secondMoment[index]
		denominator := optimizerSqrtFloat32(secondMoment[index]) + config.Epsilon
		output[index] = params[index] - config.LearningRate*gradValue/denominator
	}
}

func optimizerSqrtFloat32(value float32) float32 {
	dst := [1]float32{}
	sqrtFloat32Native(dst[:], []float32{value})

	return dst[0]
}

func runRMSpropDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return RMSpropStepFloat32(
		DefaultRMSpropConfig(),
		args[0], args[1], args[2], args[3],
	)
}
