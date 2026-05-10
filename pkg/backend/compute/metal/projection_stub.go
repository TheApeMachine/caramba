//go:build !darwin || !cgo

package metal

type ProjectionOps struct{}

func NewProjectionOps(metallib string) (*ProjectionOps, error) { return &ProjectionOps{}, nil }

func (p *ProjectionOps) Forward(shape []int, data ...[]float64) []float64 { return data[0] }

func (p *ProjectionOps) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return data[0], nil
}

func (p *ProjectionOps) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	return data[0], nil
}

func (p *ProjectionOps) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	return data[0], nil
}
