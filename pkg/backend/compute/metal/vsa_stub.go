//go:build !darwin || !cgo

package metal

/*
MetalVSAOps is the stub for non-Darwin or non-CGO builds.
All methods return errMetalUnavailable.
*/
type MetalVSAOps struct{}

/*
NewVSAOps returns an error on non-Darwin platforms.
*/
func NewVSAOps(metallib string) (*MetalVSAOps, error) {
	return nil, errMetalUnavailable
}

/*
Bind returns an error on non-Darwin platforms.
*/
func (metalVSAOps *MetalVSAOps) Bind(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

/*
Bundle returns an error on non-Darwin platforms.
*/
func (metalVSAOps *MetalVSAOps) Bundle(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}

/*
Similarity returns an error on non-Darwin platforms.
*/
func (metalVSAOps *MetalVSAOps) Similarity(shape []int, data ...[]float64) ([]float64, error) {
	return nil, errMetalUnavailable
}
