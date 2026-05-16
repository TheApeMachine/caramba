//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type MetalPositional struct{}

func NewPositional(metallib string) (*MetalPositional, error) { return nil, metalUnavailable() }

func (m *MetalPositional) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPEForwardAt(
	base float64,
	positionStart int,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPEForwardAtMode(
	base float64,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPEForwardAtModeConfig(
	config rotary.Config,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPETensor(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	base float64,
	positionStart int,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPETensorMode(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	base float64,
	positionStart int,
	mode string,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) RoPETensorModeConfig(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	config rotary.Config,
	positionStart int,
	mode string,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) ALiBiForward(shape []int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) ALiBiForwardCausal(shape []int, causal bool) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalPositional) ALiBiTensor(
	outputShape computetensor.Shape,
	causal bool,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}
