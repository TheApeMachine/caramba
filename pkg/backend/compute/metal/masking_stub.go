//go:build !darwin || !cgo

package metal

type MetalMasking struct{}
type MetalCausalMask struct{}
type MetalApplyMask struct{}

func NewMasking(metallib string) (*MetalMasking, error) { return nil, errMetalUnavailable }

func (m *MetalMasking) NewCausalMask() *MetalCausalMask {
	panic("MetalMasking.NewCausalMask: metal masking backend unavailable (darwin + cgo required)")
}

func (op *MetalCausalMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (m *MetalMasking) NewApplyMask() *MetalApplyMask {
	panic("MetalMasking.NewApplyMask: metal masking backend unavailable (darwin + cgo required)")
}

func (op *MetalApplyMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}
