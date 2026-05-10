//go:build !darwin || !cgo

package metal

type MathOps struct{}

func NewMathOps(metallib string) (*MathOps, error) { return &MathOps{}, nil }

func (m *MathOps) Matmul(shape []int, data ...[]float64) []float64         { return data[0] }
func (m *MathOps) Add(shape []int, data ...[]float64) []float64            { return data[0] }
func (m *MathOps) Mul(shape []int, data ...[]float64) []float64            { return data[0] }
func (m *MathOps) InvSqrtDimScale(shape []int, data ...[]float64) []float64 { return data[0] }
func (m *MathOps) Exp(shape []int, data ...[]float64) []float64            { return data[0] }
func (m *MathOps) Log(shape []int, data ...[]float64) []float64            { return data[0] }
func (m *MathOps) Softmax(shape []int, data ...[]float64) []float64        { return data[0] }

func (m *MathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) []float64 {
	return data[0]
}

func (m *MathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) []float64 {
	return data[0]
}
