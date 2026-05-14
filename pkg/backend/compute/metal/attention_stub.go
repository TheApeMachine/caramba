//go:build !darwin || !cgo

package metal

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

func (m *MetalAttention) GQA(q, k, v []float64, batch, numHeads, numKVHeads, seqLen, headDim int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalAttention) SlidingWindow(q, k, v []float64, batch, numHeads, seqLen, headDim, window int) ([]float64, error) {
	return nil, metalUnavailable()
}
