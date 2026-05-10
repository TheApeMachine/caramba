//go:build !darwin || !cgo

package metal

type MetalPositional struct{}

func NewPositional(metallib string) (*MetalPositional, error) { return &MetalPositional{}, nil }

func (m *MetalPositional) Forward(shape []int, data ...[]float64) []float64 { return data[0] }

func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	return data[0], nil
}

func (m *MetalPositional) ALiBiForward(shape []int) ([]float64, error) {
	return []float64{}, nil
}
