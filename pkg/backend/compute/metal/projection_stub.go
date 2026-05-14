//go:build !darwin || !cgo

package metal

type ProjectionOps struct{}

func NewProjectionOps(metallib string) (*ProjectionOps, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

func (p *ProjectionOps) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}
