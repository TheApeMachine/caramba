/*
Package embedding implements token-embedding lookup for transformer models.

The TokenEmbedding operation maps integer token IDs (passed as float64) to
dense embedding vectors drawn from a learned weight table.

Forward signature: shape=[batch, seq_len], data[0]=tokens (float64 token IDs)
Output: [batch*seq_len*DModel]
*/
package embedding

import (
	"math/rand"
)

// TokenEmbedding holds the embedding weight table and its dimensions.
type TokenEmbedding struct {
	// Weight is the flat embedding matrix, row-major: Weight[id*DModel : (id+1)*DModel]
	Weight    []float64
	VocabSize int
	DModel    int
	InitStd   float64
}

// NewTokenEmbedding creates a TokenEmbedding and initialises the weight table
// from a normal distribution N(0, initStd).  Pass initStd=0 to use the default
// value of 0.02.
func NewTokenEmbedding(vocabSize, dModel int, initStd float64) *TokenEmbedding {
	if initStd == 0 {
		initStd = 0.02
	}
	w := make([]float64, vocabSize*dModel)
	for i := range w {
		w[i] = rand.NormFloat64() * initStd
	}
	return &TokenEmbedding{
		Weight:    w,
		VocabSize: vocabSize,
		DModel:    dModel,
		InitStd:   initStd,
	}
}

// Forward looks up the embedding vector for each token ID in data[0].
//
//   shape = [batch, seq_len]
//   data[0] = flat float64 token IDs, length batch*seq_len
//
// Returns a flat []float64 of length batch*seq_len*DModel.
func (te *TokenEmbedding) Forward(shape []int, data ...[]float64) []float64 {
	tokens := data[0]
	n := len(tokens) // batch * seq_len
	out := make([]float64, n*te.DModel)
	applyLookup(out, tokens, te.Weight, te.DModel)
	return out
}
