package vsa

/*
Permute applies a deterministic cyclic shift of k positions to encode VSA roles.
Role-binding via permutation is lossless and invertible, making it ideal for
encoding positional structure in hypervector sequences.
shape=[N], data[0]=vector → out[N] shifted by k positions.
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
	n := shape[0]
	out := make([]float64, n)
	k := ((permute.k % n) + n) % n
	copy(out[k:], data[0][:n-k])
	copy(out[:k], data[0][n-k:])
	return out
}
