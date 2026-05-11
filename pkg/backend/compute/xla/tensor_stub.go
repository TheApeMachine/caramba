//go:build !cgo || !xla

package xla

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
TensorBackend is unavailable without the cgo and xla build tags.
*/
type TensorBackend struct{}

/*
NewTensorBackend reports that resident XLA tensors require cgo and xla.
*/
func NewTensorBackend(platform string) (*TensorBackend, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.XLA
}

func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return nil, fmt.Errorf("xla tensor: unavailable without cgo and xla build tags")
}

func (tensorBackend *TensorBackend) Close() error {
	return nil
}
