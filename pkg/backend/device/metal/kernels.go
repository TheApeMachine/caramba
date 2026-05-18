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
	registerBinaryFloat32Kernel("max", metalBinaryFloat32Max)
	registerBinaryFloat32Kernel("min", metalBinaryFloat32Min)
	registerBinaryFloat32Kernel("eq", metalBinaryFloat32Eq)
	registerBinaryFloat32Kernel("ne", metalBinaryFloat32Ne)
	registerBinaryFloat32Kernel("lt", metalBinaryFloat32Lt)
	registerBinaryFloat32Kernel("le", metalBinaryFloat32Le)
	registerBinaryFloat32Kernel("gt", metalBinaryFloat32Gt)
	registerBinaryFloat32Kernel("ge", metalBinaryFloat32Ge)
	registerBinaryFloat16Kernels()
	registerBinaryBFloat16Kernels()
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

func (backend *Backend) MaxFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Max, left, right)
}

func (backend *Backend) MinFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Min, left, right)
}

func (backend *Backend) EqFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Eq, left, right)
}

func (backend *Backend) NeFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Ne, left, right)
}

func (backend *Backend) LtFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Lt, left, right)
}

func (backend *Backend) LeFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Le, left, right)
}

func (backend *Backend) GtFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Gt, left, right)
}

func (backend *Backend) GeFloat32(
	ctx context.Context,
	left tensor.Tensor,
	right tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.binaryFloat32(ctx, metalBinaryFloat32Ge, left, right)
}

func registerBinaryFloat32Kernel(name string, operation metalBinaryFloat32Operation) {
	registerBinaryKernel(name, dtype.Float32, runBinaryFloat32(operation))
}

func registerBinaryFloat16Kernels() {
	registerBinaryDTypeKernels(dtype.Float16)
}

func registerBinaryBFloat16Kernels() {
	registerBinaryDTypeKernels(dtype.BFloat16)
}

func registerBinaryDTypeKernels(storageDType dtype.DType) {
	registerBinaryKernel("add", storageDType, runBinaryElementwise(metalBinaryFloat32Add))
	registerBinaryKernel("sub", storageDType, runBinaryElementwise(metalBinaryFloat32Sub))
	registerBinaryKernel("mul", storageDType, runBinaryElementwise(metalBinaryFloat32Mul))
	registerBinaryKernel("div", storageDType, runBinaryElementwise(metalBinaryFloat32Div))
	registerBinaryKernel("max", storageDType, runBinaryElementwise(metalBinaryFloat32Max))
	registerBinaryKernel("min", storageDType, runBinaryElementwise(metalBinaryFloat32Min))
	registerBinaryKernel("eq", storageDType, runBinaryElementwise(metalBinaryFloat32Eq))
	registerBinaryKernel("ne", storageDType, runBinaryElementwise(metalBinaryFloat32Ne))
	registerBinaryKernel("lt", storageDType, runBinaryElementwise(metalBinaryFloat32Lt))
	registerBinaryKernel("le", storageDType, runBinaryElementwise(metalBinaryFloat32Le))
	registerBinaryKernel("gt", storageDType, runBinaryElementwise(metalBinaryFloat32Gt))
	registerBinaryKernel("ge", storageDType, runBinaryElementwise(metalBinaryFloat32Ge))
}

func registerBinaryKernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
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

func runBinaryElementwise(operation metalBinaryFloat32Operation) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		return runMetalBinaryElementwise(operation, args[0], args[1], args[2])
	}
}
