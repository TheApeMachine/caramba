package metal

import (
	"context"
	"slices"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	registerBinaryFloat32Kernel("add", metalBinaryFloat32Add)
	registerBinaryFloat32Kernel("sub", metalBinaryFloat32Sub)
	registerBinaryFloat32Kernel("mul", metalBinaryFloat32Mul)
	registerBinaryFloat32Kernel("div", metalBinaryFloat32Div)
}

/*
AddFloat32 dispatches a real Metal compute kernel for elementwise
float32 addition and returns a Metal-resident output tensor.
*/
func (backend *Backend) AddFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Add, left, right)
}

/*
SubFloat32 dispatches a real Metal compute kernel for elementwise
float32 subtraction and returns a Metal-resident output tensor.
*/
func (backend *Backend) SubFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Sub, left, right)
}

/*
MulFloat32 dispatches a real Metal compute kernel for elementwise
float32 multiplication and returns a Metal-resident output tensor.
*/
func (backend *Backend) MulFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Mul, left, right)
}

/*
DivFloat32 dispatches a real Metal compute kernel for elementwise
float32 division and returns a Metal-resident output tensor.
*/
func (backend *Backend) DivFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Div, left, right)
}

func registerBinaryFloat32Kernel(name string, operation metalBinaryFloat32Operation) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runBinaryFloat32(operation),
	})
}

func (backend *Backend) binaryFloat32(
	ctx context.Context,
	operation metalBinaryFloat32Operation,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	if left.DType() != dtype.Float32 || right.DType() != dtype.Float32 {
		return nil, tensor.ErrDTypeMismatch
	}

	if !slices.Equal(left.Shape().Dims(), right.Shape().Dims()) {
		return nil, tensor.ErrShapeMismatch
	}

	out, err := backend.bridge.empty(left.Shape(), dtype.Float32)
	if err != nil {
		return nil, err
	}

	if err := runMetalBinaryFloat32(operation, left, right, out); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func runBinaryFloat32(operation metalBinaryFloat32Operation) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		return runMetalBinaryFloat32(operation, args[0], args[1], args[2])
	}
}
