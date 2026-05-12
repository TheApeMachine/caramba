package math

// Softmax computes softmax over the last dimension of the input.
// Each row passes through a dedicated AVX2/SSE2/NEON kernel that fuses max,
// exp (polynomial range-reduced), sum, and reciprocal-divide inline.
type Softmax struct{}

func NewSoftmax() *Softmax { return &Softmax{} }

func (op *Softmax) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dimSize := shape[len(shape)-1]
	out := make([]float64, len(x))
	copy(out, x)
	n := len(x) / dimSize

	for i := 0; i < n; i++ {
		row := out[i*dimSize : (i+1)*dimSize]
		softmaxRow(row)
	}

	return out
}

// softmaxRow dispatches to the fused SIMD kernel.
func softmaxRow(row []float64) {
	softmaxRowSIMD(row)
}
