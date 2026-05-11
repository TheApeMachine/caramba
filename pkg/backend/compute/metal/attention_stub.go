//go:build !darwin || !cgo

package metal

import "errors"

var errMetalUnavailable = errors.New("metal backend unavailable: requires darwin + cgo")

type MetalAttention struct{}

func NewAttention(metallib string) (*MetalAttention, error) {
	return nil, errMetalUnavailable
}

func (m *MetalAttention) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalAttention) SDPA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalAttention) MQA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalAttention) GQA(q, k, v []float64, batch, numHeads, numKVHeads, seqLen, headDim int) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalAttention) SlidingWindow(q, k, v []float64, batch, numHeads, seqLen, headDim, window int) ([]float64, error) {
	return nil, errMetalUnavailable
}
