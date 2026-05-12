package vsa

import "fmt"

/*
Similarity computes the dot-product cosine similarity between two VSA hypervectors.
Assumes both vectors are already L2-normalised (which Bundle guarantees), so the
dot product equals the cosine similarity directly.
shape=[N], data[0]=a, data[1]=b → out=[dot_product].
*/
type Similarity struct{}

/*
NewSimilarity instantiates a new Similarity operation.
*/
func NewSimilarity() *Similarity { return &Similarity{} }

/*
Forward returns a length-1 slice containing the dot product of data[0] and data[1].
*/
func (similarity *Similarity) Forward(shape []int, data ...[]float64) []float64 {
	if len(data) < 2 || data[0] == nil || data[1] == nil {
		panic(fmt.Sprintf("vsa: Similarity.Forward: len(data)=%d, need >= 2 with non-nil data[0], data[1]", len(data)))
	}

	na, nb := len(data[0]), len(data[1])

	if na != nb {
		panic(fmt.Sprintf(
			"vsa: Similarity.Forward before applyDot: len(data[0])=%d len(data[1])=%d must match",
			na, nb,
		))
	}

	if na == 0 {
		panic("vsa: Similarity.Forward: empty vectors not allowed before applyDot")
	}

	return []float64{applyDot(data[0], data[1])}
}
