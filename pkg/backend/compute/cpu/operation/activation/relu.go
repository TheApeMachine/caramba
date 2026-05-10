package activation

/*
ReLU applies elementwise max(0, x) using SIMD on amd64/arm64.
*/
type ReLU struct{}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (relu *ReLU) Forward(shape []int, data ...[]float64) []float64 {
	if len(data) == 0 || len(data[0]) == 0 {
		return []float64{}
	}
	input := data[0]
	out := make([]float64, len(input))
	applyReLU(out, input)
	return out
}
