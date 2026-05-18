//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

type ProjectionOps struct{}

func NewProjectionOps(metallib string) (*ProjectionOps, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) FusedQKVTensor(
	input, weight, bias computetensor.Tensor,
	outputShape computetensor.Shape,
	rows, inFeatures, outFeatures int,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}
