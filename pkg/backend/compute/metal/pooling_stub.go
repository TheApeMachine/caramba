//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

type PoolingOps struct{}
type MaxPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
}
type AvgPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
	CountIncludePad      bool
	DivisorOverride      int
}

func NewPoolingOps(metallib string) (*PoolingOps, error) { return nil, metalUnavailable() }

func (m *PoolingOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) MaxPool2d(shape []int, params MaxPool2dParams, data []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) AvgPool2d(shape []int, params AvgPool2dParams, data []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) AdaptiveAvgPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) AdaptiveMaxPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) MaxPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	params MaxPool2dParams,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) AvgPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	params AvgPool2dParams,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) AdaptiveAvgPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (m *PoolingOps) AdaptiveMaxPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}
