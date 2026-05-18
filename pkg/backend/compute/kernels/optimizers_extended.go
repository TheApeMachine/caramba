package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Optimizer kernels beyond Adam: AdamW (decoupled weight decay), Lion
(sign-of-momentum), and SGD with momentum. Adam itself lives in
optimizers.go.

Per Phase 8.4, optimizer state is fp32 master per master-precision
convention. The args order matches AdamStepFloat32: (params,
gradients, firstMoment, secondMoment, output) for the moment-based
optimizers; (params, gradients, momentum, output) for SGD-with-
momentum; (params, gradients, momentum, output) for Lion.

The configuration scalars (learning rate, betas, weight decay, etc.)
live in the *Config structs. The kernel registry's stub-runner uses
default configurations; production code calls the typed entry point
directly.
*/

type AdamWConfig struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	WeightDecay  float32
	Step         int
}

func DefaultAdamWConfig() AdamWConfig {
	return AdamWConfig{
		LearningRate: 1e-4,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  1e-2,
		Step:         1,
	}
}

type LionConfig struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	WeightDecay  float32
}

func DefaultLionConfig() LionConfig {
	return LionConfig{
		LearningRate: 1e-4,
		Beta1:        0.9,
		Beta2:        0.99,
		WeightDecay:  0,
	}
}

type SGDConfig struct {
	LearningRate float32
	Momentum     float32
	WeightDecay  float32
	Nesterov     bool
}

func DefaultSGDConfig() SGDConfig {
	return SGDConfig{
		LearningRate: 1e-2,
		Momentum:     0.9,
		WeightDecay:  0,
		Nesterov:     false,
	}
}

func init() {
	Default.Register(Kernel{
		Name: "adamw_step",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAdamWStepDefault,
	})

	Default.Register(Kernel{
		Name: "lion_step",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLionStepDefault,
	})

	Default.Register(Kernel{
		Name: "sgd_step",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSGDStepDefault,
	})
}

/*
AdamWStepFloat32 applies one AdamW update step. AdamW differs from
Adam by applying weight decay directly to the parameters (decoupled)
rather than mixing it into the gradient.
*/
func AdamWStepFloat32(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output tensor.Tensor,
) error {
	paramsView, gradView, firstView, secondView, outView, err := adamViews(
		params, gradients, firstMoment, secondMoment, output,
	)

	if err != nil {
		return err
	}

	adamWStepSlices(config, paramsView, gradView, firstView, secondView, outView)
	return nil
}

func adamWStepSlicesScalar(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))

	for index, gradValue := range gradients {
		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue
		secondMoment[index] = config.Beta2*secondMoment[index] + (1-config.Beta2)*gradValue*gradValue

		biasCorrectedFirst := firstMoment[index] / beta1Correction
		biasCorrectedSecond := secondMoment[index] / beta2Correction

		denominator := float32(math.Sqrt(float64(biasCorrectedSecond))) + config.Epsilon
		gradStep := config.LearningRate * biasCorrectedFirst / denominator
		decayStep := config.LearningRate * config.WeightDecay * params[index]

		output[index] = params[index] - gradStep - decayStep
	}
}

func adamViews(
	params, gradients, firstMoment, secondMoment, output tensor.Tensor,
) (paramsView, gradView, firstView, secondView, outView []float32, err error) {
	paramsView, err = params.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	gradView, err = gradients.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	firstView, err = firstMoment.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	secondView, err = secondMoment.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	outView, err = output.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	expectedLen := len(paramsView)

	if len(gradView) != expectedLen ||
		len(firstView) != expectedLen ||
		len(secondView) != expectedLen ||
		len(outView) != expectedLen {
		return nil, nil, nil, nil, nil, tensor.ErrShapeMismatch
	}

	return paramsView, gradView, firstView, secondView, outView, nil
}

func runAdamWStepDefault(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return AdamWStepFloat32(
		DefaultAdamWConfig(),
		args[0], args[1], args[2], args[3], args[4],
	)
}

/*
LionStepFloat32 applies one Lion update step. Lion uses sign-of-
momentum, not raw momentum: very memory-efficient (only one running
state buffer instead of two).
*/
func LionStepFloat32(
	config LionConfig,
	params, gradients, momentum, output tensor.Tensor,
) error {
	paramsView, err := params.Float32Native()

	if err != nil {
		return err
	}

	gradView, err := gradients.Float32Native()

	if err != nil {
		return err
	}

	momentumView, err := momentum.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	lionStepSlices(config, paramsView, gradView, momentumView, outView)
	return nil
}

func lionStepSlicesScalar(config LionConfig, params, gradients, momentum, output []float32) {
	for index, gradValue := range gradients {
		update := config.Beta1*momentum[index] + (1-config.Beta1)*gradValue

		var sign float32

		switch {
		case update > 0:
			sign = 1
		case update < 0:
			sign = -1
		}

		decayStep := config.WeightDecay * params[index]
		output[index] = params[index] - config.LearningRate*(sign+decayStep)

		momentum[index] = config.Beta2*momentum[index] + (1-config.Beta2)*gradValue
	}
}

func runLionStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return LionStepFloat32(
		DefaultLionConfig(),
		args[0], args[1], args[2], args[3],
	)
}

/*
SGDStepFloat32 applies SGD with optional momentum and weight decay.
Nesterov-flavor lookahead is supported via the config flag.
*/
func SGDStepFloat32(
	config SGDConfig,
	params, gradients, momentum, output tensor.Tensor,
) error {
	paramsView, err := params.Float32Native()

	if err != nil {
		return err
	}

	gradView, err := gradients.Float32Native()

	if err != nil {
		return err
	}

	momentumView, err := momentum.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	sgdStepSlices(config, paramsView, gradView, momentumView, outView)
	return nil
}

func sgdStepSlicesScalar(config SGDConfig, params, gradients, momentum, output []float32) {
	for index, gradValue := range gradients {
		effective := gradValue + config.WeightDecay*params[index]

		momentum[index] = config.Momentum*momentum[index] + effective

		update := momentum[index]

		if config.Nesterov {
			update = effective + config.Momentum*momentum[index]
		}

		output[index] = params[index] - config.LearningRate*update
	}
}

func runSGDStepDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return SGDStepFloat32(
		DefaultSGDConfig(),
		args[0], args[1], args[2], args[3],
	)
}
