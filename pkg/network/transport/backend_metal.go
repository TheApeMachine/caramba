//go:build darwin && cgo

package transport

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	devicemetal "github.com/theapemachine/caramba/pkg/backend/device/metal"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	acceleratorStreamBackendFactories = append(
		acceleratorStreamBackendFactories,
		registeredStreamBackendFactory{
			name:    "metal",
			factory: NewMetalStreamBackend,
		},
	)
}

type MetalStreamBackend struct {
	backend *devicemetal.Backend
}

func NewMetalStreamBackend() (StreamComputeBackend, error) {
	backend, err := devicemetal.NewBackend()
	if err != nil {
		return nil, err
	}

	return &MetalStreamBackend{backend: backend}, nil
}

func (backend *MetalStreamBackend) Location() tensor.Location {
	return tensor.Metal
}

func (backend *MetalStreamBackend) SupportedDTypes() []dtype.DType {
	return backend.backend.SupportedDTypes()
}

func (backend *MetalStreamBackend) SupportedLayouts() []tensor.Layout {
	return backend.backend.SupportedLayouts()
}

func (backend *MetalStreamBackend) Capabilities() tensor.Capabilities {
	return backend.backend.Capabilities()
}

func (backend *MetalStreamBackend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (tensor.Tensor, error) {
	return backend.backend.Upload(shape, sourceDType, bytes)
}

func (backend *MetalStreamBackend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (tensor.Tensor, error) {
	return backend.backend.UploadAsync(shape, sourceDType, bytes)
}

func (backend *MetalStreamBackend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return backend.backend.UploadSparse(shape, valueDType, layout, values, indices)
}

func (backend *MetalStreamBackend) Download(
	input tensor.Tensor,
) (dtype.DType, []byte, error) {
	return backend.backend.Download(input)
}

func (backend *MetalStreamBackend) Close() error {
	return backend.backend.Close()
}

func (backend *MetalStreamBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, fmt.Errorf(
		"metal stream backend: resident operation dispatch is missing for %q node %q",
		node.Op,
		node.ID,
	)
}
