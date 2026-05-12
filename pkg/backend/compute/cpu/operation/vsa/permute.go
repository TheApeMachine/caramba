package vsa

import "fmt"

/*
Permute applies a deterministic cyclic shift of k positions to encode VSA roles.
Role-binding via permutation is lossless and invertible, making it ideal for
encoding positional structure in hypervector sequences.
shape=[N], data[0]=vector → out[N] shifted by k positions.
Only data[0] is used; the variadic form matches the shared operation Forward(shape, data ...[]float64) convention.
*/
type Permute struct {
	k int
}

/*
NewPermute instantiates a new Permute operation with shift amount k.
*/
func NewPermute(k int) *Permute { return &Permute{k: k} }

/*
Forward applies a cyclic shift of k positions (wrapping).
*/
func (permute *Permute) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic("vsa: Permute.Forward: len(shape) < 1")
	}

	n := shape[0]

	if n < 0 {
		panic(fmt.Sprintf("vsa: Permute.Forward: shape[0] (n) must be non-negative, got n=%d", n))
	}

	if len(data) < 1 || data[0] == nil {
		panic(fmt.Sprintf("vsa: Permute.Forward: len(data)=%d, need >= 1 with non-nil data[0]", len(data)))
	}

	if len(data[0]) != n {
		panic(fmt.Sprintf(
			"vsa: Permute.Forward: len(data[0])=%d, need n=%d",
			len(data[0]), n,
		))
	}

	out := make([]float64, n)

	if n == 0 {
		return out
	}

	k := ((permute.k % n) + n) % n
	copy(out[k:], data[0][:n-k])
	copy(out[:k], data[0][n-k:])

	return out
}
