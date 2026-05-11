//go:build !darwin || !cgo

package metal

type ProjectionOps struct{}

func NewProjectionOps(metallib string) (*ProjectionOps, error) { return &ProjectionOps{}, nil }

func (p *ProjectionOps) Forward(shape []int, data ...[]float64) []float64 { panic(errMetalUnavailable) }

func (p *ProjectionOps) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (p *ProjectionOps) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

func (p *ProjectionOps) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}
