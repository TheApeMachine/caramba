package vsa

import (
	"fmt"
)

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
If len(data)==0, returns a zero vector of length n (no normalisation step).
*/
func (bundle *Bundle) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic("vsa: Bundle.Forward: len(shape) < 1")
	}

	n := shape[0]

	if n < 0 {
		panic(fmt.Sprintf("vsa: Bundle.Forward: shape[0] (n) must be non-negative, got n=%d", n))
	}

	if n == 0 {
		return []float64{}
	}

	for i, vec := range data {
		if len(vec) != n {
			panic(fmt.Sprintf(
				"vsa: Bundle.Forward: data[%d] len=%d, need n=%d",
				i, len(vec), n,
			))
		}
	}

	out := make([]float64, n)

	if len(data) == 0 {
		return out
	}

	for _, vec := range data {
		bundleAccum(out, vec)
	}

	applyL2Normalize(out)

	return out
}
