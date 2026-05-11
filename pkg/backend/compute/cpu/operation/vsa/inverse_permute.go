package vsa

/*
InversePermute reverses the cyclic shift applied by Permute.
Applying InversePermute(k) after Permute(k) recovers the original vector exactly,
making it the left inverse of the role-encoding operator.
shape=[N], data[0]=vector → out[N] shifted by -k positions.
*/
type InversePermute struct {
	k int
}

/*
NewInversePermute instantiates a new InversePermute operation with shift amount k.
*/
func NewInversePermute(k int) *InversePermute { return &InversePermute{k: k} }

/*
Forward reverses the cyclic shift by delegating to Permute with the complementary offset.
*/
func (inversePermute *InversePermute) Forward(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	kPos := ((inversePermute.k % n) + n) % n
	return (&Permute{k: n - kPos}).Forward(shape, data...)
}
