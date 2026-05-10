package activation

// Sigmoid applies 1/(1+exp(-x)) approximated via rational tanh elementwise.
type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (sig *Sigmoid) Forward(shape []int, data ...[]float64) []float64 {
	input := data[0]
	out := make([]float64, len(input))
	applySigmoid(out, input)
	return out
}
