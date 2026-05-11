//go:build !darwin || !cgo

package metal

type MathOps struct{}

func NewMathOps(metallib string) (*MathOps, error) {
	return nil, errMetalUnavailable
}

func (m *MathOps) Matmul(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) Add(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) Mul(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) InvSqrtDimScale(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) Exp(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) Log(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) Softmax(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}
