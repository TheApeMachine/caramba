package projection

import (
	"fmt"
	"math"
)

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
func NewTiedEmbedding(weight []float64, vocabSize, dModel int) (*TiedEmbedding, error) {
	if vocabSize <= 0 || dModel <= 0 {
		return nil, fmt.Errorf("projection: NewTiedEmbedding requires vocabSize > 0 and dModel > 0 (got %d, %d)", vocabSize, dModel)
	}

	need := int64(vocabSize) * int64(dModel)

	if need < 0 || need > int64(math.MaxInt) {
		return nil, fmt.Errorf(
			"projection: NewTiedEmbedding: vocabSize*dModel overflows int (vocabSize=%d dModel=%d)",
			vocabSize, dModel,
		)
	}

	if int64(len(weight)) < need {
		return nil, fmt.Errorf(
			"projection: NewTiedEmbedding: len(weight)=%d < vocabSize*dModel=%d",
			len(weight), need,
		)
	}

	return &TiedEmbedding{
		WeightT:   transposeF64(weight, vocabSize, dModel),
		VocabSize: vocabSize,
		DModel:    dModel,
	}, nil
}

/*
Forward computes logits = x @ WeightT.
shape = [batch, seq, DModel]; M = batch*seq.
data[0] = flattened hidden states [M * DModel].
Returns [M * VocabSize].
*/
func (te *TiedEmbedding) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic("projection: TiedEmbedding.Forward: empty shape")
	}

	last := shape[len(shape)-1]

	if last != te.DModel {
		panic(fmt.Sprintf(
			"projection: TiedEmbedding.Forward: shape[last]=%d must equal DModel=%d",
			last, te.DModel,
		))
	}

	if len(data) < 1 || data[0] == nil {
		panic("projection: TiedEmbedding.Forward: empty data")
	}

	M := 1

	for idx := 0; idx < len(shape)-1; idx++ {
		if shape[idx] < 0 {
			panic(fmt.Sprintf("projection: TiedEmbedding.Forward: shape[%d]=%d must be >= 0", idx, shape[idx]))
		}

		if shape[idx] == 0 {
			M = 0

			break
		}

		if M > math.MaxInt/shape[idx] {
			panic(fmt.Sprintf(
				"projection: TiedEmbedding.Forward: batch product overflows int (partial M=%d next=%d)",
				M, shape[idx],
			))
		}

		M *= shape[idx]
	}

	if M == 0 {
		return []float64{}
	}

	K := te.DModel
	N := te.VocabSize

	if K > 0 && M > len(data[0])/K {
		panic(fmt.Sprintf(
			"projection: TiedEmbedding.Forward: len(data[0])=%d insufficient for M=%d and DModel=%d (need len >= M*DModel)",
			len(data[0]), M, te.DModel,
		))
	}

	wantW := int64(K) * int64(N)

	if wantW < 0 || wantW > int64(math.MaxInt) || len(te.WeightT) != int(wantW) {
		panic(fmt.Sprintf(
			"projection: TiedEmbedding.Forward: len(WeightT)=%d want DModel*VocabSize=%d",
			len(te.WeightT), int(wantW),
		))
	}

	if int64(M)*int64(N) < 0 || int64(M)*int64(N) > int64(math.MaxInt) {
		panic(fmt.Sprintf(
			"projection: TiedEmbedding.Forward: M*VocabSize overflows int (M=%d VocabSize=%d)",
			M, N,
		))
	}

	out := make([]float64, M*N)
	applyMatmul(out, data[0], te.WeightT, M, K, N)

	return out
}
