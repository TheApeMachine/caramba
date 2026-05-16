//go:build !darwin || !cgo

package metal

type MetalActivation struct{}

func New(metallib string) (*MetalActivation, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) Forward(_ []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) ReLU(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) LeakyReLU(input []float64, alpha float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) GELU(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) Tanh(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) Sigmoid(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) Swish(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) SELU(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalActivation) SwiGLU(input []float64) ([]float64, error) {
	return nil, metalUnavailable()
}
