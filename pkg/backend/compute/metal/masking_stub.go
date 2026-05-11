//go:build !darwin || !cgo

package metal

type MetalMasking struct{}
type MetalCausalMask struct{}
type MetalApplyMask struct{}

func NewMasking(metallib string) (*MetalMasking, error) { return nil, errMetalUnavailable }

func (m *MetalMasking) NewCausalMask() *MetalCausalMask { return &MetalCausalMask{} }

func (op *MetalCausalMask) Forward(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}

func (m *MetalMasking) NewApplyMask() *MetalApplyMask { return &MetalApplyMask{} }

func (op *MetalApplyMask) Forward(shape []int, data ...[]float64) []float64 {
	panic(errMetalUnavailable)
}
