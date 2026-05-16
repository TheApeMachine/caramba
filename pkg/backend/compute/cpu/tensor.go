package cpu

import (
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend owns resident host tensors for CPU graph execution.
*/
type TensorBackend struct {
	storage *computetensor.HostBackend
}

/*
NewTensorBackend creates the persistent host tensor backend.
*/
func NewTensorBackend() *TensorBackend {
	return &TensorBackend{
		storage: computetensor.NewHostBackend(),
	}
}

/*
Location identifies host storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.Host
}

/*
UploadFloat64 copies host values into resident host storage.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.storage.UploadFloat64(shape, values)
}

/*
AdoptFloat64 takes ownership of a freshly produced host slice without copying.
*/
func (tensorBackend *TensorBackend) AdoptFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.storage.AdoptFloat64(shape, values)
}

/*
DownloadFloat64 returns resident host tensor data via HostBackend (zero-copy slice alias).

Independent buffers require CloneFloat64 on the tensor.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	tensor computetensor.Float64Tensor,
) ([]float64, error) {
	return tensorBackend.storage.DownloadFloat64(tensor)
}

/*
Close releases the backend storage owner.
*/
func (tensorBackend *TensorBackend) Close() error {
	return tensorBackend.storage.Close()
}
