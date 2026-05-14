//go:build !darwin || !cgo

package metal

type MetalMasking struct{}
type MetalCausalMask struct{}
type MetalApplyMask struct{}

func NewMasking(metallib string) (*MetalMasking, error) { return nil, metalUnavailable() }

func (m *MetalMasking) NewCausalMask() *MetalCausalMask {
	return &MetalCausalMask{}
}

func (op *MetalCausalMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (m *MetalMasking) NewApplyMask() *MetalApplyMask {
	return &MetalApplyMask{}
}

func (op *MetalApplyMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}
