package vsa

import "fmt"

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
	if len(shape) < 1 {
		panic("vsa: InversePermute.Forward: len(shape) < 1")
	}

	n := shape[0]

	if n < 0 {
		panic(fmt.Sprintf("vsa: InversePermute.Forward: shape[0] (n) must be non-negative, got n=%d", n))
	}

	if len(data) < 1 || data[0] == nil {
		panic(fmt.Sprintf("vsa: InversePermute.Forward: len(data)=%d, need >= 1 with non-nil data[0]", len(data)))
	}

	if len(data[0]) != n {
		panic(fmt.Sprintf(
			"vsa: InversePermute.Forward: len(data[0])=%d, need n=%d",
			len(data[0]), n,
		))
	}

	if n == 0 {
		return (&Permute{k: 0}).Forward(shape, data[0])
	}

	kPos := ((inversePermute.k % n) + n) % n

	return (&Permute{k: n - kPos}).Forward(shape, data[0])
}
