//go:build darwin && cgo

package metal

import (
	"errors"
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalActiveScalarConfig struct {
	likelihood        *metalTensor
	posterior         *metalTensor
	prior             *metalTensor
	auxiliary         *metalTensor
	first             *metalTensor
	second            *metalTensor
	third             *metalTensor
	out               *metalTensor
	scratch           *metalTensor
	elementDType      metalElementDType
	count             uint32
	stateCount        uint32
	partialCount      uint32
	statePartialCount uint32
}

type metalActiveBinaryConfig struct {
	left         *metalTensor
	right        *metalTensor
	out          *metalTensor
	scratch      *metalTensor
	elementDType metalElementDType
	count        uint32
	partialCount uint32
}

func requireMetalActiveFreeEnergy(
	likelihood tensor.Tensor,
	posterior tensor.Tensor,
	prior tensor.Tensor,
	auxiliary tensor.Tensor,
	out tensor.Tensor,
) (metalActiveScalarConfig, error) {
	tensors, err := requireMetalTensors(likelihood, posterior, prior, auxiliary, out)
	if err != nil {
		return metalActiveScalarConfig{}, err
	}

	if err := requireMetalActiveSameDTypeAndBridge(tensors...); err != nil {
		return metalActiveScalarConfig{}, err
	}

	count, err := requireMetalActiveAlignedScalar(tensors[0], tensors[1], tensors[2], tensors[4])
	if err != nil {
		return metalActiveScalarConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalActiveScalarConfig{}, err
	}

	return metalActiveScalarConfig{
		likelihood: tensors[0], posterior: tensors[1], prior: tensors[2],
		auxiliary: tensors[3], out: tensors[4], elementDType: elementDType,
		count: uint32(count), partialCount: uint32(metalActivePartialCount(count)),
	}, nil
}

func requireMetalExpectedFreeEnergy(
	predictedObs tensor.Tensor,
	preferredObs tensor.Tensor,
	predictedState tensor.Tensor,
	out tensor.Tensor,
) (metalActiveScalarConfig, error) {
	tensors, err := requireMetalTensors(predictedObs, preferredObs, predictedState, out)
	if err != nil {
		return metalActiveScalarConfig{}, err
	}

	if err := requireMetalActiveSameDTypeAndBridge(tensors...); err != nil {
		return metalActiveScalarConfig{}, err
	}

	obsCount, stateCount, err := requireMetalExpectedFreeEnergyShapes(tensors[0], tensors[1], tensors[2], tensors[3])
	if err != nil {
		return metalActiveScalarConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalActiveScalarConfig{}, err
	}

	obsPartial := metalActivePartialCount(obsCount)
	statePartial := metalActivePartialCount(stateCount)
	return metalActiveScalarConfig{
		first: tensors[0], second: tensors[1], third: tensors[2], out: tensors[3],
		elementDType: elementDType, count: uint32(obsCount), stateCount: uint32(stateCount),
		partialCount: uint32(obsPartial), statePartialCount: uint32(statePartial),
	}, nil
}

func requireMetalActiveBinary(
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) (metalActiveBinaryConfig, error) {
	tensors, err := requireMetalTensors(left, right, out)
	if err != nil {
		return metalActiveBinaryConfig{}, err
	}

	if err := requireMetalActiveSameDTypeAndBridge(tensors...); err != nil {
		return metalActiveBinaryConfig{}, err
	}

	if !tensors[0].shape.Equal(tensors[1].shape) ||
		!tensors[0].shape.Equal(tensors[2].shape) ||
		tensors[0].shape.Len() > math.MaxUint32 {
		return metalActiveBinaryConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalActiveBinaryConfig{}, err
	}

	count := tensors[0].shape.Len()
	return metalActiveBinaryConfig{
		left: tensors[0], right: tensors[1], out: tensors[2], elementDType: elementDType,
		count: uint32(count), partialCount: uint32(metalActivePartialCount(count)),
	}, nil
}

func requireMetalActiveAlignedScalar(
	first *metalTensor,
	second *metalTensor,
	third *metalTensor,
	out *metalTensor,
) (int, error) {
	if !first.shape.Equal(second.shape) || !first.shape.Equal(third.shape) || out.shape.Len() < 1 {
		return 0, tensor.ErrShapeMismatch
	}

	if first.shape.Len() == 0 || first.shape.Len() > math.MaxUint32 {
		return 0, tensor.ErrShapeMismatch
	}

	return first.shape.Len(), nil
}

func requireMetalExpectedFreeEnergyShapes(
	predictedObs *metalTensor,
	preferredObs *metalTensor,
	predictedState *metalTensor,
	out *metalTensor,
) (int, int, error) {
	if !predictedObs.shape.Equal(preferredObs.shape) || out.shape.Len() < 1 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	obsCount := predictedObs.shape.Len()
	stateCount := predictedState.shape.Len()
	if obsCount == 0 || stateCount == 0 || obsCount > math.MaxUint32 || stateCount > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return obsCount, stateCount, nil
}

func (config *metalActiveScalarConfig) allocateScratch() error {
	partialCount := int(config.partialCount + config.statePartialCount)
	scratch, err := newMetalActiveScratch(config.out.bridge, partialCount)
	if err != nil {
		return err
	}

	config.scratch = scratch
	return nil
}

func (config *metalActiveBinaryConfig) allocateScratch() error {
	scratch, err := newMetalActiveScratch(config.out.bridge, int(config.partialCount))
	if err != nil {
		return err
	}

	config.scratch = scratch
	return nil
}

func (config metalActiveScalarConfig) beginCompletion(sources ...*metalTensor) (uint64, error) {
	return metalCompletions.BeginMany([]*metalTensor{config.out, config.scratch}, sources...)
}

func (config metalActiveBinaryConfig) beginCompletion(sources ...*metalTensor) (uint64, error) {
	return metalCompletions.BeginMany([]*metalTensor{config.out, config.scratch}, sources...)
}

func (config metalActiveScalarConfig) closeScratch() {
	_ = config.scratch.Close()
}

func (config metalActiveBinaryConfig) closeScratch() {
	_ = config.scratch.Close()
}

func requireMetalActiveSameDTypeAndBridge(tensors ...*metalTensor) error {
	if len(tensors) == 0 {
		return tensor.ErrShapeMismatch
	}

	storageDType := tensors[0].dtype
	bridge := tensors[0].bridge

	for _, target := range tensors[1:] {
		if target.dtype != storageDType {
			return tensor.ErrDTypeMismatch
		}

		if target.bridge != bridge {
			return errors.New("metal active inference: tensors belong to different Metal backends")
		}
	}

	return nil
}
