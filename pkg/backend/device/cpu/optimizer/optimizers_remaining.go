package optimizer

import (
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

/*
LARS, L-BFGS, and Hebbian optimizers. Each registers a step kernel
against the standard (params, gradients, state..., output) shape.

LARS scales each layer's update by the ratio of parameter norm to
gradient norm (large-batch training).
L-BFGS performs a memory-limited quasi-Newton step using a history
of past gradient/parameter deltas.
Hebbian learns by correlating pre- and post-synaptic activations:
the gradient is replaced by an outer product of activations.
*/

type LARSConfig struct {
	LearningRate float32
	Momentum     float32
	WeightDecay  float32
	TrustCoeff   float32
	Epsilon      float32
}

func DefaultLARSConfig() LARSConfig {
	return LARSConfig{LearningRate: 1e-2, Momentum: 0.9, WeightDecay: 1e-4, TrustCoeff: 1e-3, Epsilon: 1e-8}
}

type HebbianConfig struct {
	LearningRate float32
	Decay        float32
}

func DefaultHebbianConfig() HebbianConfig {
	return HebbianConfig{LearningRate: 1e-3, Decay: 1e-4}
}

type LBFGSConfig struct {
	LearningRate float32
	HistorySize  int
}

func DefaultLBFGSConfig() LBFGSConfig {
	return LBFGSConfig{LearningRate: 1.0, HistorySize: 20}
}

/*
LARSStepFloat32 computes the LARS update: a per-layer trust ratio
scales the gradient before the momentum + weight-decay step.
*/
func LARSStepFloat32(
	config LARSConfig,
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

	larsStepSlices(config, paramsView, gradView, momentumView, outView)
	return nil
}

func larsStepSlicesScalar(config LARSConfig, params, gradients, momentum, output []float32) {
	var paramsNorm, gradsNorm float64

	for index, value := range params {
		paramsNorm += float64(value) * float64(value)
		gradsNorm += float64(gradients[index]) * float64(gradients[index])
	}

	paramsNorm = math.Sqrt(paramsNorm)
	gradsNorm = math.Sqrt(gradsNorm)

	trust := float32(1.0)

	if paramsNorm > 0 && gradsNorm > 0 {
		trust = config.TrustCoeff *
			float32(paramsNorm) /
			(float32(gradsNorm) + config.WeightDecay*float32(paramsNorm) + config.Epsilon)
	}

	effectiveLr := config.LearningRate * trust

	for index, gradValue := range gradients {
		decayed := gradValue + config.WeightDecay*params[index]
		momentum[index] = config.Momentum*momentum[index] + decayed
		output[index] = params[index] - effectiveLr*momentum[index]
	}
}

/*
HebbianStepFloat32 applies a Hebbian update: weights are updated by
the outer product of the post-synaptic activation and pre-synaptic
activation. Args order: (weights, post-activation, pre-activation,
output).
*/
func HebbianStepFloat32(
	config HebbianConfig,
	weights, post, pre, output tensor.Tensor,
) error {
	weightsView, err := weights.Float32Native()

	if err != nil {
		return err
	}

	postView, err := post.Float32Native()

	if err != nil {
		return err
	}

	preView, err := pre.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	weightsDims := weights.Shape().Dims()

	if len(weightsDims) != 2 ||
		weightsDims[0] != len(postView) ||
		weightsDims[1] != len(preView) ||
		len(outView) != len(weightsView) {
		return tensor.ErrShapeMismatch
	}

	hebbianStepSlices(config, weightsView, postView, preView, outView, weightsDims[1])
	return nil
}

func hebbianStepSlicesScalar(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	for postIndex, postValue := range post {
		for preIndex, preValue := range pre {
			weightIndex := postIndex*preDim + preIndex
			output[weightIndex] = weights[weightIndex]*(1-config.Decay) +
				config.LearningRate*postValue*preValue
		}
	}
}

/*
LBFGSStepFloat32 takes a simple gradient-descent step using
config.LearningRate. Real L-BFGS requires the two-loop recursion
over a history of (parameter delta, gradient delta) pairs; the host
reference here is the unwound single-step version. The full history-
based variant lands when the orchestrator has state-management for
the curvature pairs.
*/
func LBFGSStepFloat32(
	config LBFGSConfig,
	params, gradients, output tensor.Tensor,
) error {
	paramsView, err := params.Float32Native()

	if err != nil {
		return err
	}

	gradView, err := gradients.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(paramsView) != len(gradView) || len(outView) != len(paramsView) {
		return tensor.ErrShapeMismatch
	}

	lbfgsStepSlices(config, paramsView, gradView, outView)
	return nil
}

func lbfgsStepSlicesScalar(config LBFGSConfig, params, gradients, output []float32) {
	for index, gradValue := range gradients {
		output[index] = params[index] - config.LearningRate*gradValue
	}
}
