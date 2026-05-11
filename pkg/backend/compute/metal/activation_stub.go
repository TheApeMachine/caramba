//go:build !darwin || !cgo

package metal

type MetalActivation struct{}

func New(metallib string) (*MetalActivation, error) {
	return nil, errMetalUnavailable
}

func (m *MetalActivation) Forward(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MetalActivation) ReLU(input []float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalActivation) LeakyReLU(input []float64, alpha float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalActivation) GELU(input []float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalActivation) Tanh(input []float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalActivation) Sigmoid(input []float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalActivation) SwiGLU(input []float64) ([]float64, error) {
	return nil, errMetalUnavailable
}
