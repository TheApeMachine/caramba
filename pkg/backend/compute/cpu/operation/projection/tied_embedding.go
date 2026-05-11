package projection

/*
TiedEmbedding projects hidden states back to vocabulary logits using the
transposed token embedding matrix.

WeightT is the pre-transposed form [DModel × VocabSize] so Forward performs
one matmul with no per-call allocation. The caller owns the original
[VocabSize × DModel] weight; this type caches its transpose.
*/
type TiedEmbedding struct {
	WeightT   []float64 // [DModel × VocabSize] pre-transposed
	VocabSize int
	DModel    int
}

/*
NewTiedEmbedding creates a TiedEmbedding projection.
weight must already be initialised (shared with the token embedding table),
layout [VocabSize × DModel] row-major.
*/
func NewTiedEmbedding(weight []float64, vocabSize, dModel int) *TiedEmbedding {
	return &TiedEmbedding{
		WeightT:   transposeF64(weight, vocabSize, dModel),
		VocabSize: vocabSize,
		DModel:    dModel,
	}
}

/*
Forward computes logits = x @ WeightT.
shape = [batch, seq, DModel]; M = batch*seq.
data[0] = flattened hidden states [M * DModel].
Returns [M * VocabSize].
*/
func (te *TiedEmbedding) Forward(shape []int, data ...[]float64) []float64 {
	M := 1
	for i := 0; i < len(shape)-1; i++ {
		M *= shape[i]
	}

	K := te.DModel
	N := te.VocabSize
	out := make([]float64, M*N)
	applyMatmul(out, data[0], te.WeightT, M, K, N)

	return out
}
