//go:build !darwin || !cgo

package metal

type MetalPositional struct{}

func NewPositional(metallib string) (*MetalPositional, error) { return nil, errMetalUnavailable }

func (m *MetalPositional) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalPositional) ALiBiForward(shape []int) ([]float64, error) {
	return nil, errMetalUnavailable
}
