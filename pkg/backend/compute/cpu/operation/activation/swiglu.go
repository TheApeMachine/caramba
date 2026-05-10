package activation

// SwiGLU expects input of length 2n (gates|values) and returns length n.
// Each output[i] = sigmoid(gates[i]) * values[i].
type SwiGLU struct{}

func NewSwiGLU() *SwiGLU {
	return &SwiGLU{}
}

func (swi *SwiGLU) Forward(shape []int, data ...[]float64) []float64 {
	input := data[0]
	n := shape[len(shape)-1]
	out := make([]float64, n)
	applySwiGLU(out, input)
	return out
}
