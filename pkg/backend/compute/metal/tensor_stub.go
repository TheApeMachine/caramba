//go:build !darwin || !cgo

package metal

import computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend reports unavailable Metal tensor support when Metal is not built in.
*/
type TensorBackend struct{}

/*
NewTensorBackend creates a Metal tensor backend stub.
*/
func NewTensorBackend() (*TensorBackend, error) {
	return &TensorBackend{}, metalUnavailable()
}

/*
Location identifies the intended Metal storage location.
*/
func (*TensorBackend) Location() computetensor.Location {
	return computetensor.Metal
}

/*
UploadFloat64 rejects uploads when Metal support is not built in.
*/
func (*TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return nil, metalUnavailable()
}

/*
DownloadFloat64 rejects downloads when Metal support is not built in.
*/
func (*TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return nil, metalUnavailable()
}

/*
Close is a no-op for the Metal tensor backend stub.
*/
func (*TensorBackend) Close() error {
	return nil
}
