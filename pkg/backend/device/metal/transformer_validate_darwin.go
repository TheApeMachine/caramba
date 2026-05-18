//go:build darwin && cgo

package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalEmbeddingConfig struct {
	table        *metalTensor
	indices      *metalTensor
	offsets      *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	vocab        uint32
	hidden       uint32
	indexCount   uint32
	bagCount     uint32
}

type metalMaskConfig struct {
	input        *metalTensor
	mask         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
	rows         uint32
	cols         uint32
}

type metalALiBiConfig struct {
	scores       *metalTensor
	slope        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	cols         uint32
}

func requireMetalEmbeddingLookup(
	table tensor.Tensor,
	indices tensor.Tensor,
	out tensor.Tensor,
) (metalEmbeddingConfig, error) {
	config, err := metalEmbeddingLookupTensors(table, indices, out)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	vocab, hidden, indexCount, err := metalEmbeddingLookupDims(config)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.table.dtype)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	config.vocab = uint32(vocab)
	config.hidden = uint32(hidden)
	config.indexCount = uint32(indexCount)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalEmbeddingBag(
	table tensor.Tensor,
	indices tensor.Tensor,
	offsets tensor.Tensor,
	out tensor.Tensor,
) (metalEmbeddingConfig, error) {
	config, err := metalEmbeddingBagTensors(table, indices, offsets, out)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	vocab, hidden, indexCount, bagCount, err := metalEmbeddingBagDims(config)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.table.dtype)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	config.vocab = uint32(vocab)
	config.hidden = uint32(hidden)
	config.indexCount = uint32(indexCount)
	config.bagCount = uint32(bagCount)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalApplyMask(
	input tensor.Tensor,
	mask tensor.Tensor,
	out tensor.Tensor,
) (metalMaskConfig, error) {
	inputTensor, maskTensor, outTensor, err := requireMetalSameDType3(input, mask, out)
	if err != nil {
		return metalMaskConfig{}, err
	}

	if !inputTensor.shape.Equal(maskTensor.shape) || !inputTensor.shape.Equal(outTensor.shape) {
		return metalMaskConfig{}, tensor.ErrShapeMismatch
	}

	if inputTensor.shape.Len() > math.MaxUint32 {
		return metalMaskConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalMaskConfig{}, err
	}

	return metalMaskConfig{
		input: inputTensor, mask: maskTensor, out: outTensor,
		elementDType: elementDType, count: uint32(inputTensor.shape.Len()),
	}, nil
}

func requireMetalCausalMask(
	input tensor.Tensor,
	out tensor.Tensor,
) (metalMaskConfig, error) {
	inputTensor, outTensor, err := requireMetalShapeSameDType(input, out)
	if err != nil {
		return metalMaskConfig{}, err
	}

	dims := outTensor.shape.Dims()
	if len(dims) != 2 {
		return metalMaskConfig{}, tensor.ErrShapeMismatch
	}

	if inputTensor.shape.Len() == 0 || outTensor.shape.Len() > math.MaxUint32 {
		return metalMaskConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(outTensor.dtype)
	if err != nil {
		return metalMaskConfig{}, err
	}

	return metalMaskConfig{
		input: inputTensor, out: outTensor, elementDType: elementDType,
		rows: uint32(dims[0]), cols: uint32(dims[1]),
	}, nil
}

func requireMetalALiBiBias(
	scores tensor.Tensor,
	slope tensor.Tensor,
	out tensor.Tensor,
) (metalALiBiConfig, error) {
	scoresTensor, slopeTensor, outTensor, err := requireMetalSameDType3(scores, slope, out)
	if err != nil {
		return metalALiBiConfig{}, err
	}

	dims := scoresTensor.shape.Dims()
	if len(dims) != 2 || slopeTensor.shape.Len() < 1 || !scoresTensor.shape.Equal(outTensor.shape) {
		return metalALiBiConfig{}, tensor.ErrShapeMismatch
	}

	if scoresTensor.shape.Len() > math.MaxUint32 {
		return metalALiBiConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(scoresTensor.dtype)
	if err != nil {
		return metalALiBiConfig{}, err
	}

	return metalALiBiConfig{
		scores: scoresTensor, slope: slopeTensor, out: outTensor,
		elementDType: elementDType, rows: uint32(dims[0]), cols: uint32(dims[1]),
	}, nil
}

func metalEmbeddingLookupTensors(
	table tensor.Tensor,
	indices tensor.Tensor,
	out tensor.Tensor,
) (metalEmbeddingConfig, error) {
	tableTensor, err := requireMetalTensor(table)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	indicesTensor, err := requireMetalTensor(indices)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	if err := requireEmbeddingTensorTypes(tableTensor, indicesTensor, nil, outTensor); err != nil {
		return metalEmbeddingConfig{}, err
	}

	return metalEmbeddingConfig{table: tableTensor, indices: indicesTensor, out: outTensor}, nil
}

func metalEmbeddingBagTensors(
	table tensor.Tensor,
	indices tensor.Tensor,
	offsets tensor.Tensor,
	out tensor.Tensor,
) (metalEmbeddingConfig, error) {
	config, err := metalEmbeddingLookupTensors(table, indices, out)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	offsetsTensor, err := requireMetalTensor(offsets)
	if err != nil {
		return metalEmbeddingConfig{}, err
	}

	if err := requireEmbeddingTensorTypes(config.table, config.indices, offsetsTensor, config.out); err != nil {
		return metalEmbeddingConfig{}, err
	}

	config.offsets = offsetsTensor
	return config, nil
}

func requireEmbeddingTensorTypes(
	table *metalTensor,
	indices *metalTensor,
	offsets *metalTensor,
	out *metalTensor,
) error {
	if table.dtype != out.dtype || indices.dtype != dtype.Int32 {
		return tensor.ErrDTypeMismatch
	}

	if offsets != nil && offsets.dtype != dtype.Int32 {
		return tensor.ErrDTypeMismatch
	}

	if table.bridge != indices.bridge || table.bridge != out.bridge {
		return tensor.ErrShapeMismatch
	}

	if offsets != nil && offsets.bridge != table.bridge {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func metalEmbeddingLookupDims(config metalEmbeddingConfig) (int, int, int, error) {
	tableDims := config.table.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(tableDims) != 2 || len(outDims) != 2 || len(config.indices.shape.Dims()) != 1 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	vocab, hidden, indexCount := tableDims[0], tableDims[1], config.indices.shape.Len()
	if outDims[0] != indexCount || outDims[1] != hidden {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return vocab, hidden, indexCount, requireTransformerUint32(vocab, hidden, indexCount)
}

func metalEmbeddingBagDims(config metalEmbeddingConfig) (int, int, int, int, error) {
	tableDims := config.table.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(tableDims) != 2 || len(outDims) != 2 ||
		len(config.indices.shape.Dims()) != 1 || len(config.offsets.shape.Dims()) != 1 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	vocab, hidden := tableDims[0], tableDims[1]
	indexCount, bagCount := config.indices.shape.Len(), config.offsets.shape.Len()
	if outDims[0] != bagCount || outDims[1] != hidden {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	err := requireTransformerUint32(vocab, hidden, indexCount, bagCount)
	return vocab, hidden, indexCount, bagCount, err
}

func requireTransformerUint32(values ...int) error {
	for _, value := range values {
		if value < 0 || int64(value) > math.MaxUint32 {
			return tensor.ErrShapeMismatch
		}
	}

	return nil
}
