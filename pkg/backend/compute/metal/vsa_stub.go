//go:build !darwin || !cgo

package metal

/*
MetalVSAOps is the stub for non-Darwin or non-CGO builds.
All methods return metalUnavailable().
*/
type MetalVSAOps struct{}

/*
NewVSAOps returns an error on non-Darwin platforms.
*/
func NewVSAOps(metallib string) (*MetalVSAOps, error) {
	return nil, metalUnavailable()
}

/*
Bind returns an error on non-Darwin platforms.
*/
func (metalVSAOps *MetalVSAOps) Bind(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

/*
Bundle returns an error on non-Darwin platforms.
*/
func (metalVSAOps *MetalVSAOps) Bundle(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}

/*
Similarity returns an error on non-Darwin platforms.
*/
func (metalVSAOps *MetalVSAOps) Similarity(shape []int, data ...[]float64) ([]float64, error) {
	return nil, metalUnavailable()
}
