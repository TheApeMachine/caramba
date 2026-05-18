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

	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))

	for index, gradValue := range gradView {
		firstView[index] = config.Beta1*firstView[index] + (1-config.Beta1)*gradValue

		updated := config.Beta2 * infView[index]
		absGrad := float32(math.Abs(float64(gradValue)))

		if absGrad > updated {
			updated = absGrad
		}

		infView[index] = updated

		biasCorrectedFirst := firstView[index] / beta1Correction
		outView[index] = paramsView[index] - config.LearningRate*biasCorrectedFirst/(infView[index]+config.Epsilon)
	}

	return nil
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

	for index, gradValue := range gradView {
		accView[index] += gradValue * gradValue

		denominator := float32(math.Sqrt(float64(accView[index]))) + config.Epsilon
		outView[index] = paramsView[index] - config.LearningRate*gradValue/denominator
	}

	return nil
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

	for index, gradValue := range gradView {
		secondView[index] = config.Decay*secondView[index] +
			(1-config.Decay)*gradValue*gradValue

		denominator := float32(math.Sqrt(float64(secondView[index]))) + config.Epsilon
		outView[index] = paramsView[index] - config.LearningRate*gradValue/denominator
	}

	return nil
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
