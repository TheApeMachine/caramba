//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

type MetalPositional struct{}

func NewPositional(metallib string) (*MetalPositional, error) { return nil, metalUnavailable() }

func (m *MetalPositional) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPETensor(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	base float64,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) ALiBiForward(shape []int) ([]float64, error) {
	return nil, metalUnavailable()
}
