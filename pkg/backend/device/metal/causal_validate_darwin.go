//go:build darwin && cgo

package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalCausalBinaryConfig struct {
	first        *metalTensor
	second       *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
	rows         uint32
	inner        uint32
	cols         uint32
}

type metalCausalTernaryConfig struct {
	first        *metalTensor
	second       *metalTensor
	third        *metalTensor
	fourth       *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
	rows         uint32
	inner        uint32
	cols         uint32
}

type metalCausalScalarConfig struct {
	first        *metalTensor
	second       *metalTensor
	third        *metalTensor
	out          *metalTensor
	scratch      *metalTensor
	elementDType metalElementDType
	count        uint32
	partialCount uint32
}

func requireMetalBackdoorAdjustment(
	conditional tensor.Tensor,
	marginal tensor.Tensor,
	out tensor.Tensor,
) (metalCausalBinaryConfig, error) {
	config, err := requireMetalCausalBinary(conditional, marginal, out)
	if err != nil {
		return metalCausalBinaryConfig{}, err
	}

	conditionDims := config.first.shape.Dims()
	if len(conditionDims) != 3 {
		return metalCausalBinaryConfig{}, tensor.ErrShapeMismatch
	}

	xCount, zCount, yCount := conditionDims[0], conditionDims[1], conditionDims[2]
	if config.second.shape.Len() != zCount || !config.out.shape.Equal(mustMetalShape(xCount, yCount)) {
		return metalCausalBinaryConfig{}, tensor.ErrShapeMismatch
	}

	return config.withDims(xCount, zCount, yCount)
}

func requireMetalFrontdoorAdjustment(
	mediator tensor.Tensor,
	outcome tensor.Tensor,
	marginal tensor.Tensor,
	out tensor.Tensor,
) (metalCausalTernaryConfig, error) {
	config, err := requireMetalCausalTernary(mediator, outcome, marginal, out)
	if err != nil {
		return metalCausalTernaryConfig{}, err
	}

	mediatorDims := config.first.shape.Dims()
	outcomeDims := config.second.shape.Dims()
	if len(mediatorDims) != 2 || len(outcomeDims) != 3 {
		return metalCausalTernaryConfig{}, tensor.ErrShapeMismatch
	}

	xCount, mCount, yCount := mediatorDims[0], mediatorDims[1], outcomeDims[2]
	if outcomeDims[0] != xCount || outcomeDims[1] != mCount ||
		config.third.shape.Len() != xCount || !config.out.shape.Equal(mustMetalShape(xCount, yCount)) {
		return metalCausalTernaryConfig{}, tensor.ErrShapeMismatch
	}

	return config.withDims(xCount, mCount, yCount)
}

func requireMetalDoIntervene(
	adjacency tensor.Tensor,
	intervened tensor.Tensor,
	out tensor.Tensor,
) (metalCausalBinaryConfig, error) {
	config, err := requireMetalCausalInt32(adjacency, intervened, out)
	if err != nil {
		return metalCausalBinaryConfig{}, err
	}

	dims := config.first.shape.Dims()
	if len(dims) != 2 || dims[0] != dims[1] || !config.out.shape.Equal(config.first.shape) {
		return metalCausalBinaryConfig{}, tensor.ErrShapeMismatch
	}

	return config.withDims(dims[0], config.second.shape.Len(), dims[1])
}

func requireMetalCATE(
	treated tensor.Tensor,
	control tensor.Tensor,
	out tensor.Tensor,
) (metalCausalBinaryConfig, error) {
	config, err := requireMetalCausalBinary(treated, control, out)
	if err != nil {
		return metalCausalBinaryConfig{}, err
	}

	if !config.first.shape.Equal(config.second.shape) || !config.first.shape.Equal(config.out.shape) {
		return metalCausalBinaryConfig{}, tensor.ErrShapeMismatch
	}

	return config.withCount(config.out.shape.Len())
}

func requireMetalCounterfactual(
	observedY tensor.Tensor,
	observedX tensor.Tensor,
	counterfactualX tensor.Tensor,
	slope tensor.Tensor,
	out tensor.Tensor,
) (metalCausalTernaryConfig, error) {
	tensors, err := requireMetalTensors(observedY, observedX, counterfactualX, slope, out)
	if err != nil {
		return metalCausalTernaryConfig{}, err
	}

	if err := requireMetalCausalSameDTypeAndBridge(tensors...); err != nil {
		return metalCausalTernaryConfig{}, err
	}

	if !tensors[0].shape.Equal(tensors[1].shape) || !tensors[0].shape.Equal(tensors[2].shape) ||
		!tensors[0].shape.Equal(tensors[4].shape) || tensors[3].shape.Len() < 1 {
		return metalCausalTernaryConfig{}, tensor.ErrShapeMismatch
	}

	config, err := newMetalCausalTernaryConfig(tensors[0], tensors[1], tensors[2], tensors[4])
	if err != nil {
		return metalCausalTernaryConfig{}, err
	}

	config.fourth = tensors[3]
	return config.withCount(tensors[0].shape.Len())
}

func requireMetalIVEstimate(
	instrument tensor.Tensor,
	treatment tensor.Tensor,
	outcome tensor.Tensor,
	out tensor.Tensor,
) (metalCausalScalarConfig, error) {
	config, err := requireMetalCausalScalar(instrument, treatment, outcome, out)
	if err != nil {
		return metalCausalScalarConfig{}, err
	}

	if !config.first.shape.Equal(config.second.shape) || !config.first.shape.Equal(config.third.shape) ||
		config.out.shape.Len() < 1 || config.first.shape.Len() < 2 {
		return metalCausalScalarConfig{}, tensor.ErrShapeMismatch
	}

	return config.withCount(config.first.shape.Len())
}

func requireMetalDAGMarkovFactorization(
	conditionals tensor.Tensor,
	parents tensor.Tensor,
	out tensor.Tensor,
) (metalCausalScalarConfig, error) {
	config, err := requireMetalCausalScalarInt32(conditionals, parents, out)
	if err != nil {
		return metalCausalScalarConfig{}, err
	}

	if config.first.shape.Len() == 0 || config.out.shape.Len() < 1 {
		return metalCausalScalarConfig{}, tensor.ErrShapeMismatch
	}

	return config.withCount(config.first.shape.Len())
}

func requireMetalCausalBinary(
	first tensor.Tensor,
	second tensor.Tensor,
	out tensor.Tensor,
) (metalCausalBinaryConfig, error) {
	tensors, err := requireMetalTensors(first, second, out)
	if err != nil {
		return metalCausalBinaryConfig{}, err
	}

	if err := requireMetalCausalSameDTypeAndBridge(tensors...); err != nil {
		return metalCausalBinaryConfig{}, err
	}

	return newMetalCausalBinaryConfig(tensors[0], tensors[1], tensors[2])
}

func requireMetalCausalScalar(
	first tensor.Tensor,
	second tensor.Tensor,
	third tensor.Tensor,
	out tensor.Tensor,
) (metalCausalScalarConfig, error) {
	tensors, err := requireMetalTensors(first, second, third, out)
	if err != nil {
		return metalCausalScalarConfig{}, err
	}

	if err := requireMetalCausalSameDTypeAndBridge(tensors...); err != nil {
		return metalCausalScalarConfig{}, err
	}

	return newMetalCausalScalarConfig(tensors[0], tensors[1], tensors[2], tensors[3])
}

func requireMetalCausalScalarInt32(
	first tensor.Tensor,
	second tensor.Tensor,
	out tensor.Tensor,
) (metalCausalScalarConfig, error) {
	tensors, err := requireMetalTensors(first, second, out)
	if err != nil {
		return metalCausalScalarConfig{}, err
	}

	if err := requireMetalCausalSameDTypeInt32Bridge(tensors[0], tensors[1], tensors[2]); err != nil {
		return metalCausalScalarConfig{}, err
	}

	return newMetalCausalScalarConfig(tensors[0], tensors[1], nil, tensors[2])
}

func requireMetalCausalTernary(
	first tensor.Tensor,
	second tensor.Tensor,
	third tensor.Tensor,
	out tensor.Tensor,
) (metalCausalTernaryConfig, error) {
	tensors, err := requireMetalTensors(first, second, third, out)
	if err != nil {
		return metalCausalTernaryConfig{}, err
	}

	if err := requireMetalCausalSameDTypeAndBridge(tensors...); err != nil {
		return metalCausalTernaryConfig{}, err
	}

	return newMetalCausalTernaryConfig(tensors[0], tensors[1], tensors[2], tensors[3])
}

func requireMetalCausalInt32(
	first tensor.Tensor,
	second tensor.Tensor,
	out tensor.Tensor,
) (metalCausalBinaryConfig, error) {
	tensors, err := requireMetalTensors(first, second, out)
	if err != nil {
		return metalCausalBinaryConfig{}, err
	}

	if err := requireMetalCausalSameDTypeInt32Bridge(tensors[0], tensors[1], tensors[2]); err != nil {
		return metalCausalBinaryConfig{}, err
	}

	return newMetalCausalBinaryConfig(tensors[0], tensors[1], tensors[2])
}
