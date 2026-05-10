package math

// Add performs elementwise addition: out[i] = data[0][i] + data[1][i].
type Add struct{}

func NewAdd() *Add { return &Add{} }

func (op *Add) Forward(shape []int, data ...[]float64) []float64 {
	out := make([]float64, len(data[0]))
	addVec(out, data[0], data[1])
	return out
}
