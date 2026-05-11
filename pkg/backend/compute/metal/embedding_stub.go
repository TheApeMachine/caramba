//go:build !darwin || !cgo

package metal

type EmbeddingOps struct{}

func NewEmbeddingOps(metallib string, vocabSize, dModel int) (*EmbeddingOps, error) {
	return nil, errMetalUnavailable
}

func (e *EmbeddingOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (e *EmbeddingOps) TokenEmbedding(tokens []float64, weight []float64) ([]float64, error) {
	return nil, errMetalUnavailable
}
