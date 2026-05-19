package optimizer

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Adam optimizer step. State is fp32 master per Phase 8.4 of
TENSOR_BACKEND_REWRITE.md. Args order:

	(params, gradients, firstMoment, secondMoment, output)

firstMoment and secondMoment are running estimates updated in place;
output is the new params (which may alias params for in-place
optimization).

LearningRate, Beta1, Beta2, Epsilon, and the timestep are passed
through the AdamConfig parameter on the helper function rather than
as tensors, because they are scalars that don't fit the
elementwise-tensor dispatch model. The orchestrator binds them at
plan time.
*/
type AdamConfig struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	Step         int
}

/*
DefaultAdamConfig returns the standard transformer training hyper-
parameters: lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8.
*/
func DefaultAdamConfig() AdamConfig {
	return AdamConfig{
		LearningRate: 1e-4,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Step:         1,
	}
}

/*
AdamStepFloat32 applies one Adam update step elementwise on fp32
storage. Params, gradients, firstMoment, secondMoment, and output
must all have matching shape and Float32 dtype.
*/
func AdamStepFloat32(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output tensor.Tensor,
) error {
	paramsView, err := params.Float32Native()

	if err != nil {
		return err
	}

	gradView, err := gradients.Float32Native()

	if err != nil {
		return err
	}

	firstView, err := firstMoment.Float32Native()

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
		len(paramsView) != len(firstView) ||
		len(paramsView) != len(secondView) ||
		len(paramsView) != len(outView) {
		return tensor.ErrShapeMismatch
	}

	adamStepSlices(config, paramsView, gradView, firstView, secondView, outView)
	return nil
}

/*
adamStepSlicesScalar is the portable scalar reference. The production
adamStepSlices dispatches to a NEON-backed variant on arm64
(AdamStepSlicesNEON in optimizers_f32_dispatch_arm64.go). On other
architectures, adamStepSlices = adamStepSlicesScalar.
*/
func adamStepSlicesScalar(
	config AdamConfig,
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
		output[index] = params[index] - config.LearningRate*biasCorrectedFirst/denominator
	}
}
