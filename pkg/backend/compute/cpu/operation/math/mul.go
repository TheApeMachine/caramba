package math

// Mul performs elementwise multiply: out[i] = data[0][i] * data[1][i].
type Mul struct{}

func NewMul() *Mul { return &Mul{} }

func (op *Mul) Forward(shape []int, data ...[]float64) []float64 {
	out := make([]float64, len(data[0]))
	mulVec(out, data[0], data[1])
	return out
}
