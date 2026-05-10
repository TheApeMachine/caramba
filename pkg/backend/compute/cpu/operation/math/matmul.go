package math

/*
Matmul performs matrix multiplication A [M*K] x B [K*N] -> C [M*N].
shape = [M, K, N].
*/
type Matmul struct{}

func NewMatmul() *Matmul { return &Matmul{} }

func (op *Matmul) Forward(shape []int, data ...[]float64) []float64 {
	M, K, N := shape[0], shape[1], shape[2]
	out := make([]float64, M*N)
	applyMatmul(out, data[0], data[1], M, K, N)
	return out
}
