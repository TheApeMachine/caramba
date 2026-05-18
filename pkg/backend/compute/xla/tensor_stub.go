//go:build !cgo || !xla

package xla

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var _ computetensor.Backend = (*TensorBackend)(nil)
var _ executor.Backend = (*TensorBackend)(nil)

/*
TensorBackend is unavailable without the cgo and xla build tags.
*/
type TensorBackend struct{}

/*
NewTensorBackend reports that resident XLA tensors require cgo and xla.

platform is accepted so call signatures match the cgo+xla implementation.
*/
func NewTensorBackend(platform string) (*TensorBackend, error) {
	_ = platform

	return &TensorBackend{}, xlaUnavailable()
}

func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.XLA
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

	return nil, xlaUnavailable()
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

	return nil, xlaUnavailable()
}

func (tensorBackend *TensorBackend) Download(
	input computetensor.Tensor,
) (dtype.DType, []byte, error) {
	_ = input

	return dtype.Invalid, nil, xlaUnavailable()
}

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

	return nil, xlaUnavailable()
}

func xlaUnavailable() error {
	return fmt.Errorf("%w: xla tensor: unavailable without cgo and xla build tags", computetensor.ErrNeedsPlatformSetup)
}
