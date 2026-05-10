package math

/*
Outer computes the outer product of two vectors: out[i*N+j] = a[i] * b[j].
shape: [M, N] where M = len(data[0]), N = len(data[1]).
Used to accumulate Hebbian weight matrices: W += outer(p, p) for each stored pattern p.
*/
type Outer struct{}

func NewOuter() *Outer { return &Outer{} }

func (op *Outer) Forward(shape []int, data ...[]float64) []float64 {
	M, N := shape[0], shape[1]
	a, b := data[0], data[1]
	out := make([]float64, M*N)

	for row := range M {
		outerRow(out[row*N:(row+1)*N], b, a[row])
	}

	return out
}
