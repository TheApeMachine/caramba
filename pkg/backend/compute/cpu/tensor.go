package cpu

import (
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
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

func (tensorBackend *TensorBackend) SupportedDTypes() []dtype.DType {
	return tensorBackend.storage.SupportedDTypes()
}

func (tensorBackend *TensorBackend) SupportedLayouts() []computetensor.Layout {
	return tensorBackend.storage.SupportedLayouts()
}

func (tensorBackend *TensorBackend) Capabilities() computetensor.Capabilities {
	return tensorBackend.storage.Capabilities()
}

func (tensorBackend *TensorBackend) Upload(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (computetensor.Tensor, error) {
	return tensorBackend.storage.Upload(shape, sourceDType, bytes)
}

func (tensorBackend *TensorBackend) UploadAsync(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (computetensor.Tensor, error) {
	return tensorBackend.storage.UploadAsync(shape, sourceDType, bytes)
}

func (tensorBackend *TensorBackend) UploadSparse(
	shape computetensor.Shape,
	valueDType dtype.DType,
	layout computetensor.Layout,
	values []byte,
	indices []computetensor.SparseIndex,
) (computetensor.SparseTensor, error) {
	return tensorBackend.storage.UploadSparse(shape, valueDType, layout, values, indices)
}

func (tensorBackend *TensorBackend) Download(
	input computetensor.Tensor,
) (dtype.DType, []byte, error) {
	return tensorBackend.storage.Download(input)
}

/*
Close releases the backend storage owner.
*/
func (tensorBackend *TensorBackend) Close() error {
	return tensorBackend.storage.Close()
}
