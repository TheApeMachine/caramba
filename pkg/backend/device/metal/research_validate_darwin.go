//go:build darwin && cgo

package metal

import (
	"errors"
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalResearchUnaryConfig struct {
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalResearchBinaryConfig struct {
	left         *metalTensor
	right        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalPCMatrixConfig struct {
	weights      *metalTensor
	state        *metalTensor
	error        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	outCount     uint32
	inCount      uint32
}

func requireMetalResearchUnary(
	input tensor.Tensor,
	out tensor.Tensor,
) (metalResearchUnaryConfig, error) {
	inputTensor, outTensor, err := requireMetalResearchSameDType(input, out)
	if err != nil {
		return metalResearchUnaryConfig{}, err
	}

	if !inputTensor.shape.Equal(outTensor.shape) || inputTensor.shape.Len() > math.MaxUint32 {
		return metalResearchUnaryConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalResearchUnaryConfig{}, err
	}

	return metalResearchUnaryConfig{
		input: inputTensor, out: outTensor, elementDType: elementDType,
		count: uint32(inputTensor.shape.Len()),
	}, nil
}

func requireMetalResearchBinary(
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) (metalResearchBinaryConfig, error) {
	tensors, err := requireMetalTensors(left, right, out)
	if err != nil {
		return metalResearchBinaryConfig{}, err
	}

	if err := requireMetalResearchSameDTypeAndBridge(tensors...); err != nil {
		return metalResearchBinaryConfig{}, err
	}

	if !tensors[0].shape.Equal(tensors[1].shape) ||
		!tensors[0].shape.Equal(tensors[2].shape) ||
		tensors[0].shape.Len() > math.MaxUint32 {
		return metalResearchBinaryConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalResearchBinaryConfig{}, err
	}

	return metalResearchBinaryConfig{
		left: tensors[0], right: tensors[1], out: tensors[2],
		elementDType: elementDType, count: uint32(tensors[0].shape.Len()),
	}, nil
}

func requireMetalPCPrediction(
	weights tensor.Tensor,
	state tensor.Tensor,
	out tensor.Tensor,
) (metalPCMatrixConfig, error) {
	tensors, err := requireMetalTensors(weights, state, out)
	if err != nil {
		return metalPCMatrixConfig{}, err
	}

	outCount, inCount, err := requireMetalPCMatrix(tensors[0], tensors[1], nil, tensors[2], false)
	if err != nil {
		return metalPCMatrixConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalPCMatrixConfig{}, err
	}

	return metalPCMatrixConfig{
		weights: tensors[0], state: tensors[1], out: tensors[2],
		elementDType: elementDType, outCount: outCount, inCount: inCount,
	}, nil
}

func requireMetalPCUpdate(
	weights tensor.Tensor,
	state tensor.Tensor,
	predictionError tensor.Tensor,
	out tensor.Tensor,
	weightOutput bool,
) (metalPCMatrixConfig, error) {
	tensors, err := requireMetalTensors(weights, state, predictionError, out)
	if err != nil {
		return metalPCMatrixConfig{}, err
	}

	outCount, inCount, err := requireMetalPCMatrix(
		tensors[0],
		tensors[1],
		tensors[2],
		tensors[3],
		weightOutput,
	)
	if err != nil {
		return metalPCMatrixConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalPCMatrixConfig{}, err
	}

	return metalPCMatrixConfig{
		weights: tensors[0], state: tensors[1], error: tensors[2], out: tensors[3],
		elementDType: elementDType, outCount: outCount, inCount: inCount,
	}, nil
}

func requireMetalPCMatrix(
	weights *metalTensor,
	state *metalTensor,
	predictionError *metalTensor,
	out *metalTensor,
	weightOutput bool,
) (uint32, uint32, error) {
	if err := requireMetalResearchSameDTypeAndBridge(compactMetalTensors(
		weights, state, predictionError, out,
	)...); err != nil {
		return 0, 0, err
	}

	weightDims := weights.shape.Dims()
	if len(weightDims) != 2 || weightDims[0] > math.MaxUint32 || weightDims[1] > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	outCount, inCount := weightDims[0], weightDims[1]
	if int64(outCount)*int64(inCount) > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	if state.shape.Len() != inCount {
		return 0, 0, tensor.ErrShapeMismatch
	}

	if err := requireMetalPCOutput(predictionError, out, outCount, inCount, weightOutput); err != nil {
		return 0, 0, err
	}

	return uint32(outCount), uint32(inCount), nil
}

func requireMetalPCOutput(
	predictionError *metalTensor,
	out *metalTensor,
	outCount int,
	inCount int,
	weightOutput bool,
) error {
	if predictionError != nil && predictionError.shape.Len() != outCount {
		return tensor.ErrShapeMismatch
	}

	if weightOutput && out.shape.Len() != outCount*inCount {
		return tensor.ErrShapeMismatch
	}

	if !weightOutput && out.shape.Len() != inCount && predictionError != nil {
		return tensor.ErrShapeMismatch
	}

	if !weightOutput && out.shape.Len() != outCount && predictionError == nil {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireMetalResearchSameDType(first tensor.Tensor, second tensor.Tensor) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(first, second)
	if err != nil {
		return nil, nil, err
	}

	if err := requireMetalResearchSameDTypeAndBridge(tensors...); err != nil {
		return nil, nil, err
	}

	return tensors[0], tensors[1], nil
}

func requireMetalResearchSameDTypeAndBridge(tensors ...*metalTensor) error {
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
			return errors.New("metal research: tensors belong to different Metal backends")
		}
	}

	return nil
}

func compactMetalTensors(tensors ...*metalTensor) []*metalTensor {
	out := make([]*metalTensor, 0, len(tensors))

	for _, target := range tensors {
		if target == nil {
			continue
		}

		out = append(out, target)
	}

	return out
}
