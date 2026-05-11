//go:build !linux || !cgo || !cuda

package cuda

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

const cudaTensorUnavailableMsg = "CUDA tensor unavailable: rebuild on Linux with CGO enabled and build tags linux,cgo,cuda"

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
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

/*
DownloadFloat64 rejects downloads when CUDA support is not built in.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return nil, fmt.Errorf("%s", cudaTensorUnavailableMsg)
}

/*
Close is a no-op for the CUDA tensor backend stub.
*/
func (tensorBackend *TensorBackend) Close() error {
	return nil
}
