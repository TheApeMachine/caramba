//go:build !darwin || !cgo

package metal

type MetalActivation struct{}

func New() *MetalActivation {
	return &MetalActivation{}
}

func (m *MetalActivation) Forward(shape []int, data ...[]float64) []float64 {
	return data[0]
}

func (m *MetalActivation) ReLU(input []float64) ([]float64, error) {
	return input, nil
}

func (m *MetalActivation) LeakyReLU(input []float64, alpha float64) ([]float64, error) {
	return input, nil
}

func (m *MetalActivation) GELU(input []float64) ([]float64, error) {
	return input, nil
}

func (m *MetalActivation) Tanh(input []float64) ([]float64, error) {
	return input, nil
}

func (m *MetalActivation) Sigmoid(input []float64) ([]float64, error) {
	return input, nil
}

func (m *MetalActivation) SwiGLU(input []float64) ([]float64, error) {
	return input, nil
}
