package activation

// LeakyReLU applies max(alpha*x, x) elementwise using SIMD on amd64/arm64.
type LeakyReLU struct {
	alpha float64
}

func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{alpha: alpha}
}

func (leaky *LeakyReLU) Forward(shape []int, data ...[]float64) []float64 {
	input := data[0]
	out := make([]float64, len(input))
	applyLeakyReLU(out, input, leaky.alpha)
	return out
}
