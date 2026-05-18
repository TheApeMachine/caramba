//go:build !linux || !cgo || !cuda

package cuda

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

const cudaTensorUnavailableMsg = "CUDA tensor unavailable: rebuild on Linux with CGO enabled and build tags linux,cgo,cuda"

var _ computetensor.Backend = (*TensorBackend)(nil)
var _ executor.Backend = (*TensorBackend)(nil)

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

func (tensorBackend *TensorBackend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{dtype.Float32}
}

func (tensorBackend *TensorBackend) SupportedLayouts() []computetensor.Layout {
	return []computetensor.Layout{computetensor.LayoutDense}
}

func (tensorBackend *TensorBackend) Capabilities() computetensor.Capabilities {
	return computetensor.Capabilities{
		MaxBytes:        computetensor.MaxBytesUnlimited,
		NativeAlignment: 128,
		NUMANodes:       1,
	}
}

func (tensorBackend *TensorBackend) Upload(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (computetensor.Tensor, error) {
	_ = shape
	_ = sourceDType
	_ = bytes

	return nil, cudaUnavailable()
}

func (tensorBackend *TensorBackend) UploadAsync(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (computetensor.Tensor, error) {
	return tensorBackend.Upload(shape, sourceDType, bytes)
}

func (tensorBackend *TensorBackend) UploadSparse(
	shape computetensor.Shape,
	valueDType dtype.DType,
	layout computetensor.Layout,
	values []byte,
	indices []computetensor.SparseIndex,
) (computetensor.SparseTensor, error) {
	_ = shape
	_ = valueDType
	_ = layout
	_ = values
	_ = indices

	return nil, cudaUnavailable()
}

func (tensorBackend *TensorBackend) Download(
	input computetensor.Tensor,
) (dtype.DType, []byte, error) {
	_ = input

	return dtype.Invalid, nil, cudaUnavailable()
}

/*
Close is a no-op for the CUDA tensor backend stub.
*/
func (tensorBackend *TensorBackend) Close() error {
	return nil
}

func (tensorBackend *TensorBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	_ = node
	_ = inputs

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return nil, cudaUnavailable()
}

func cudaUnavailable() error {
	return fmt.Errorf("%w: %s", computetensor.ErrNeedsPlatformSetup, cudaTensorUnavailableMsg)
}
