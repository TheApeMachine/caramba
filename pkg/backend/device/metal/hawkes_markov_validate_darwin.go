//go:build darwin && cgo

package metal

import (
	"errors"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalHawkesScalarConfig struct {
	events       *metalTensor
	first        *metalTensor
	second       *metalTensor
	third        *metalTensor
	fourth       *metalTensor
	out          *metalTensor
	scratch      *metalTensor
	elementDType metalElementDType
	eventCount   uint32
	outputCount  uint32
	partialCount uint32
}

type metalMarkovMatrixConfig struct {
	matrix       *metalTensor
	labels       *metalTensor
	out          *metalTensor
	scratch      *metalTensor
	elementDType metalElementDType
	rows         uint32
	cols         uint32
	partialCount uint32
}

func requireMetalHawkesIntensity(
	events tensor.Tensor,
	queryTimes tensor.Tensor,
	baseline tensor.Tensor,
	alpha tensor.Tensor,
	beta tensor.Tensor,
	out tensor.Tensor,
) (metalHawkesScalarConfig, error) {
	config, err := requireMetalHawkesFiveFloat(events, queryTimes, baseline, alpha, beta, out)
	if err != nil {
		return metalHawkesScalarConfig{}, err
	}

	if config.out.shape.Len() != config.first.shape.Len() {
		return metalHawkesScalarConfig{}, tensor.ErrShapeMismatch
	}

	config.outputCount = uint32(config.first.shape.Len())
	return config, nil
}

func requireMetalHawkesKernelMatrix(
	events tensor.Tensor,
	alpha tensor.Tensor,
	beta tensor.Tensor,
	out tensor.Tensor,
) (metalHawkesScalarConfig, error) {
	tensors, err := requireMetalTensors(events, alpha, beta, out)
	if err != nil {
		return metalHawkesScalarConfig{}, err
	}

	if err := requireMetalHawkesSameDTypeAndBridge(tensors...); err != nil {
		return metalHawkesScalarConfig{}, err
	}

	if tensors[1].shape.Len() < 1 || tensors[2].shape.Len() < 1 {
		return metalHawkesScalarConfig{}, tensor.ErrShapeMismatch
	}

	eventCount := tensors[0].shape.Len()
	if eventCount > math.MaxUint32 || eventCount*eventCount != tensors[3].shape.Len() {
		return metalHawkesScalarConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalHawkesScalarConfig{}, err
	}

	return metalHawkesScalarConfig{
		events: tensors[0], first: tensors[1], second: tensors[2], out: tensors[3],
		elementDType: elementDType, eventCount: uint32(eventCount),
	}, nil
}

func requireMetalHawkesLogLikelihood(
	events tensor.Tensor,
	totalTime tensor.Tensor,
	baseline tensor.Tensor,
	alpha tensor.Tensor,
	beta tensor.Tensor,
	out tensor.Tensor,
) (metalHawkesScalarConfig, error) {
	config, err := requireMetalHawkesFiveFloat(events, totalTime, baseline, alpha, beta, out)
	if err != nil {
		return metalHawkesScalarConfig{}, err
	}

	if config.out.shape.Len() < 1 || config.events.shape.Len() == 0 {
		return metalHawkesScalarConfig{}, tensor.ErrShapeMismatch
	}

	config.partialCount = uint32(metalHawkesMarkovPartialCount(config.events.shape.Len()))
	return config, nil
}

func requireMetalMarkovMutualInformation(
	joint tensor.Tensor,
	out tensor.Tensor,
) (metalMarkovMatrixConfig, error) {
	matrix, output, err := requireMetalHawkesSameFloatPair(joint, out)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	rows, cols, err := requireMetalHawkesMatrixDims(matrix)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	if output.shape.Len() < 1 {
		return metalMarkovMatrixConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(matrix.dtype)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	return metalMarkovMatrixConfig{
		matrix: matrix, out: output, elementDType: elementDType,
		rows: uint32(rows), cols: uint32(cols),
		partialCount: uint32(metalHawkesMarkovPartialCount(rows * cols)),
	}, nil
}

func requireMetalMarkovBlanketPartition(
	adjacency tensor.Tensor,
	internal tensor.Tensor,
	out tensor.Tensor,
) (metalMarkovMatrixConfig, error) {
	tensors, err := requireMetalTensors(adjacency, internal, out)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	if tensors[1].dtype != dtype.Int32 || tensors[2].dtype != dtype.Int32 {
		return metalMarkovMatrixConfig{}, tensor.ErrDTypeMismatch
	}

	if tensors[0].bridge != tensors[1].bridge || tensors[0].bridge != tensors[2].bridge {
		return metalMarkovMatrixConfig{}, errors.New("metal hawkes-markov: tensors belong to different Metal backends")
	}

	rows, cols, err := requireMetalHawkesMatrixDims(tensors[0])
	if err != nil || rows != cols || tensors[2].shape.Len() != rows {
		return metalMarkovMatrixConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	return metalMarkovMatrixConfig{
		matrix: tensors[0], labels: tensors[1], out: tensors[2],
		elementDType: elementDType, rows: uint32(rows), cols: uint32(tensors[1].shape.Len()),
	}, nil
}

func requireMetalMarkovFlow(
	mi tensor.Tensor,
	partition tensor.Tensor,
	out tensor.Tensor,
) (metalMarkovMatrixConfig, error) {
	tensors, err := requireMetalTensors(mi, partition, out)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	if tensors[1].dtype != dtype.Int32 || tensors[0].dtype != tensors[2].dtype {
		return metalMarkovMatrixConfig{}, tensor.ErrDTypeMismatch
	}

	if tensors[0].bridge != tensors[1].bridge || tensors[0].bridge != tensors[2].bridge {
		return metalMarkovMatrixConfig{}, errors.New("metal hawkes-markov: tensors belong to different Metal backends")
	}

	rows, cols, err := requireMetalHawkesMatrixDims(tensors[0])
	if err != nil || rows != cols || tensors[1].shape.Len() != rows || tensors[2].shape.Len() != rows {
		return metalMarkovMatrixConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalMarkovMatrixConfig{}, err
	}

	return metalMarkovMatrixConfig{
		matrix: tensors[0], labels: tensors[1], out: tensors[2],
		elementDType: elementDType, rows: uint32(rows),
	}, nil
}

func requireMetalHawkesFiveFloat(
	events tensor.Tensor,
	first tensor.Tensor,
	second tensor.Tensor,
	third tensor.Tensor,
	fourth tensor.Tensor,
	out tensor.Tensor,
) (metalHawkesScalarConfig, error) {
	tensors, err := requireMetalTensors(events, first, second, third, fourth, out)
	if err != nil {
		return metalHawkesScalarConfig{}, err
	}

	if err := requireMetalHawkesSameDTypeAndBridge(tensors...); err != nil {
		return metalHawkesScalarConfig{}, err
	}

	if tensors[2].shape.Len() < 1 || tensors[3].shape.Len() < 1 || tensors[4].shape.Len() < 1 ||
		tensors[0].shape.Len() > math.MaxUint32 || tensors[1].shape.Len() > math.MaxUint32 {
		return metalHawkesScalarConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalHawkesScalarConfig{}, err
	}

	return metalHawkesScalarConfig{
		events: tensors[0], first: tensors[1], second: tensors[2], third: tensors[3],
		fourth: tensors[4], out: tensors[5], elementDType: elementDType,
		eventCount: uint32(tensors[0].shape.Len()),
	}, nil
}

func requireMetalHawkesSameFloatPair(
	first tensor.Tensor,
	second tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(first, second)
	if err != nil {
		return nil, nil, err
	}

	if err := requireMetalHawkesSameDTypeAndBridge(tensors...); err != nil {
		return nil, nil, err
	}

	return tensors[0], tensors[1], nil
}

func requireMetalHawkesSameDTypeAndBridge(tensors ...*metalTensor) error {
	storageDType := tensors[0].dtype
	bridge := tensors[0].bridge

	for _, target := range tensors[1:] {
		if target.dtype != storageDType {
			return tensor.ErrDTypeMismatch
		}

		if target.bridge != bridge {
			return errors.New("metal hawkes-markov: tensors belong to different Metal backends")
		}
	}

	return nil
}

func requireMetalHawkesMatrixDims(matrix *metalTensor) (int, int, error) {
	dims := matrix.shape.Dims()
	if len(dims) != 2 || dims[0] == 0 || dims[1] == 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	if dims[0] > math.MaxUint32 || dims[1] > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return dims[0], dims[1], nil
}
