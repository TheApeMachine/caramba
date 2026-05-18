//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

type MetalAttention struct{}

func NewAttention(metallib string) (*MetalAttention, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) SDPA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) MQA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) MQATensor(
	q, k, v computetensor.Tensor,
	outputShape computetensor.Shape,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) GQA(
	q, k, v []float64,
	batch, numHeads, numKVHeads, seqLen, headDim int,
	causal bool,
) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) GQATensor(
	q, k, v computetensor.Tensor,
	outputShape computetensor.Shape,
	batch, numHeads, numKVHeads, queryLen, keyValueLen, keyValueStride, headDim int,
	causal bool,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) SlidingWindow(q, k, v []float64, batch, numHeads, seqLen, headDim, window int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) SlidingWindowTensor(
	q, k, v computetensor.Tensor,
	outputShape computetensor.Shape,
	batch, numHeads, seqLen, headDim, window int,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}
