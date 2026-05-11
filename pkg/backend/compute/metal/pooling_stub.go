//go:build !darwin || !cgo

package metal

type PoolingOps struct{}
type MaxPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
}
type AvgPool2dParams struct {
	KernelH, KernelW     int
	StrideH, StrideW     int
	PadH, PadW           int
	DilationH, DilationW int
	CeilMode             bool
	CountIncludePad      bool
}

func NewPoolingOps(metallib string) (*PoolingOps, error) { return &PoolingOps{}, nil }

// Forward is a stub when Metal is unavailable: returns the first input unchanged (passthrough)
// so graphs can load without Metal. Use the darwin+cgo build for real pooling.
func (m *PoolingOps) Forward(shape []int, data ...[]float64) []float64 {
	if len(data) == 0 {
		return nil
	}

	return data[0]
}

func (m *PoolingOps) MaxPool2d(shape []int, params MaxPool2dParams, data []float64) ([]float64, error) {
	return data, nil
}

func (m *PoolingOps) AvgPool2d(shape []int, params AvgPool2dParams, data []float64) ([]float64, error) {
	return data, nil
}

func (m *PoolingOps) AdaptiveAvgPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	return data, nil
}

func (m *PoolingOps) AdaptiveMaxPool2d(shape []int, outH, outW int, data []float64) ([]float64, error) {
	return data, nil
}
