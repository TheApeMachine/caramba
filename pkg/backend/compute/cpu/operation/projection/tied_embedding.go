package projection

/*
TiedEmbedding projects hidden states back to vocabulary logits using the
transposed token embedding matrix.

Weight layout: [VocabSize * DModel] row-major (same tensor as token embedding).
Forward:
  - shape = [batch, seq, DModel]  (shape[0]*shape[1] = M rows)
  - data[0] = x  (hidden states, flattened [M * DModel])
  - output = x @ Weight^T = [M * VocabSize]  (logits)
*/
type TiedEmbedding struct {
	Weight    []float64
	VocabSize int
	DModel    int
}

// NewTiedEmbedding creates a TiedEmbedding projection.
// Weight must already be initialised (shared with the token embedding table).
func NewTiedEmbedding(weight []float64, vocabSize, dModel int) *TiedEmbedding {
	return &TiedEmbedding{
		Weight:    weight,
		VocabSize: vocabSize,
		DModel:    dModel,
	}
}

// Forward computes logits = x @ Weight^T.
// shape = [batch, seq, DModel]; M = batch*seq.
// data[0] = flattened hidden states [M * DModel].
// Returns [M * VocabSize].
func (t *TiedEmbedding) Forward(shape []int, data ...[]float64) []float64 {
	// shape may be [batch, seq, DModel] or [M, DModel].
	M := 1
	for i := 0; i < len(shape)-1; i++ {
		M *= shape[i]
	}
	K := t.DModel
	N := t.VocabSize
	out := make([]float64, M*N)
	// Weight is [N, K]; Weight^T is [K, N].
	wT := transposeF64(t.Weight, N, K)
	applyMatmul(out, data[0], wT, M, K, N)
	return out
}
