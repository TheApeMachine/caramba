//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

type MetalShapeOps struct{}

func NewShapeOps(metallib string) (*MetalShapeOps, error) { return nil, metalUnavailable() }

func (m *MetalShapeOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) Transpose(shape []int, dim0, dim1 int, data []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) Copy(input []float64) ([]float64, error) { return nil, metalUnavailable() }
func (m *MetalShapeOps) Concat(a, b []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) UpsampleNearest2D(
	input []float64,
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) ViewAsHeads(input []float64, B, T, H, headDim int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) MergeHeads(input []float64, B, H, T, headDim int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) LastToken(input []float64, outer, sequenceLength, featureLength int) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalShapeOps) UpsampleNearest2DTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) (computetensor.Tensor, error) {
	return nil, metalUnavailable()
}
