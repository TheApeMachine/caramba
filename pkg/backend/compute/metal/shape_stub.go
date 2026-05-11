//go:build !darwin || !cgo

package metal

type MetalShapeOps struct{}

func NewShapeOps(metallib string) (*MetalShapeOps, error) { return &MetalShapeOps{}, nil }

func (m *MetalShapeOps) Forward(shape []int, data ...[]float64) []float64 { panic(errMetalUnavailable) }

func (m *MetalShapeOps) Transpose(shape []int, dim0, dim1 int, data []float64) ([]float64, error) {
	return data, nil
}

func (m *MetalShapeOps) Copy(input []float64) ([]float64, error)        { return input, nil }
func (m *MetalShapeOps) Concat(a, b []float64) ([]float64, error)       { return append(a, b...), nil }

func (m *MetalShapeOps) ViewAsHeads(input []float64, B, T, H, headDim int) ([]float64, error) {
	return input, nil
}

func (m *MetalShapeOps) MergeHeads(input []float64, B, H, T, headDim int) ([]float64, error) {
	return input, nil
}
