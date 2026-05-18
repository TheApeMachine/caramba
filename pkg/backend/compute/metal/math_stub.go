//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

type MathOps struct{}

func NewMathOps(metallib string) (*MathOps, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Matmul(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Add(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Mul(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) AddTensor(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) MulTensor(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) MatmulTensor(
	left, right computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) MatmulAddTensor(
	left, right, bias computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) MatmulAddGELUTensor(
	left, right, bias computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) InvSqrtDimScale(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) InvSqrtDimScaleTensor(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Exp(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) ExpTensor(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Log(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) LogTensor(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Softmax(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) SoftmaxTensor(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) LogSumExpTensor(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) LayerNorm(shape []int, eps float64, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) RMSNorm(shape []int, eps float64, weight []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) GroupNorm(
	shape []int,
	eps float64,
	groups int,
	weight,
	bias []float64,
	data ...[]float64,
) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) GroupNormTensor(
	input, weight, bias computetensor.Tensor,
	groups int,
	eps float64,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Sign(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) SignTensor(
	input computetensor.Tensor,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) DropoutTensor(
	input computetensor.Tensor,
	probability float64,
	training bool,
	seed int,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) Outer(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MathOps) OuterTensor(
	left computetensor.Tensor,
	right computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}
