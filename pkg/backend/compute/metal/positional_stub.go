//go:build !darwin || !cgo

package metal

type MetalPositional struct{}

func NewPositional(metallib string) (*MetalPositional, error) { return &MetalPositional{}, nil }

func (m *MetalPositional) Forward(shape []int, data ...[]float64) []float64 { panic(errMetalUnavailable) }

func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	panic(errMetalUnavailable)
}

func (m *MetalPositional) ALiBiForward(shape []int) ([]float64, error) {
	panic(errMetalUnavailable)
}
