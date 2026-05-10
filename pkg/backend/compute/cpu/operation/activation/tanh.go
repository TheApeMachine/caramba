package activation

// Tanh applies the rational-approximation tanh elementwise using SIMD on amd64/arm64.
type Tanh struct{}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (tan *Tanh) Forward(shape []int, data ...[]float64) []float64 {
	input := data[0]
	out := make([]float64, len(input))
	applyTanh(out, input)
	return out
}
