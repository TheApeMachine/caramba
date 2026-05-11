//go:build !linux || !cgo || !cuda

package cuda

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend reports unavailable CUDA tensor support when CUDA is not built in.
*/
type TensorBackend struct{}

/*
NewTensorBackend creates a CUDA tensor backend stub.
*/
func NewTensorBackend() *TensorBackend {
	return &TensorBackend{}
}

/*
Location identifies the intended CUDA storage location.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.CUDA
}

/*
UploadFloat64 rejects uploads when CUDA support is not built in.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("cuda tensor: build with linux,cgo,cuda tags")
}

/*
DownloadFloat64 rejects downloads when CUDA support is not built in.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return nil, fmt.Errorf("cuda tensor: build with linux,cgo,cuda tags")
}

/*
Close is a no-op for the CUDA tensor backend stub.
*/
func (tensorBackend *TensorBackend) Close() error {
	return nil
}
