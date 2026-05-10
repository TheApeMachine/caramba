//go:build !darwin || !cgo

package metal

type ConvolutionOps struct{}

func NewConvolutionOps(metallib string) (*ConvolutionOps, error) { return &ConvolutionOps{}, nil }

func (m *ConvolutionOps) Forward(shape []int, data ...[]float64) []float64 { return data[0] }

func (m *ConvolutionOps) Conv1d(x []float64, N, InC, L int, weight, bias []float64, OutC, K, stride, pad, dilation, groups, LOut int) ([]float64, error) {
	return x, nil
}

func (m *ConvolutionOps) Conv2d(x []float64, N, InC, H, W int, weight, bias []float64, OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int) ([]float64, error) {
	return x, nil
}

func (m *ConvolutionOps) Conv3d(x []float64, N, InC, D, H, W int, weight, bias []float64, OutC, KD, KH, KW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups int) ([]float64, error) {
	return x, nil
}

func (m *ConvolutionOps) ConvTranspose2d(x []float64, N, InC, H, W int, weight, bias []float64, OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout int) ([]float64, error) {
	return x, nil
}
