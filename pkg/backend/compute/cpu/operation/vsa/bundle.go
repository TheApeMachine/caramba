package vsa

import "math"

/*
Bundle superimposes multiple VSA hypervectors by elementwise addition followed by
L2-normalisation. The result has unit norm and lies close to all input vectors
in proportion to their contribution — the core memory-superposition primitive.
shape=[N], data[0..k-1] are the k vectors to bundle → out[N].
*/
type Bundle struct{}

/*
NewBundle instantiates a new Bundle operation.
*/
func NewBundle() *Bundle { return &Bundle{} }

/*
Forward sums all input vectors then L2-normalises the result.
*/
func (bundle *Bundle) Forward(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, n)

	for _, vec := range data {
		applyAddInPlace(out, vec)
	}

	norm := math.Sqrt(applyReduceSumSq(out))

	if norm > 1e-12 {
		applyMulScalar(out, 1.0/norm)
	}

	return out
}
