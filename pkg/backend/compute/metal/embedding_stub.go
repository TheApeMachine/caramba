//go:build !darwin || !cgo

package metal

type EmbeddingOps struct{}

func NewEmbeddingOps(metallib string, vocabSize, dModel int) (*EmbeddingOps, error) {
	return &EmbeddingOps{}, nil
}

func (e *EmbeddingOps) Forward(shape []int, data ...[]float64) []float64 { return data[0] }

func (e *EmbeddingOps) TokenEmbedding(tokens []float64, weight []float64) ([]float64, error) {
	return tokens, nil
}
