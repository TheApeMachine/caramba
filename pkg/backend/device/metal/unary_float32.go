package metal

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func init() {
	registerUnaryFloat32Kernel("relu", metalUnaryFloat32Relu)
	registerUnaryFloat32Kernel("abs", metalUnaryFloat32Abs)
	registerUnaryFloat32Kernel("neg", metalUnaryFloat32Neg)
	registerUnaryFloat32Kernel("square", metalUnaryFloat32Square)
	registerUnaryFloat32Kernel("recip", metalUnaryFloat32Recip)
	registerUnaryFloat32Kernel("sqrt", metalUnaryFloat32Sqrt)
	registerUnaryFloat32Kernel("sign", metalUnaryFloat32Sign)
	registerUnaryFloat16Kernels()
	registerUnaryBFloat16Kernels()
}

func (backend *Backend) ReluFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Relu, input)
}

func (backend *Backend) AbsFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Abs, input)
}

func (backend *Backend) NegFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Neg, input)
}

func (backend *Backend) SquareFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Square, input)
}

func (backend *Backend) RecipFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Recip, input)
}

func (backend *Backend) SqrtFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Sqrt, input)
}

func (backend *Backend) SignFloat32(
	ctx context.Context,
	input tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.unaryFloat32(ctx, metalUnaryFloat32Sign, input)
}

func registerUnaryFloat32Kernel(name string, operation metalUnaryFloat32Operation) {
	registerUnaryKernel(name, dtype.Float32, runUnaryFloat32(operation))
}

func registerUnaryFloat16Kernels() {
	registerUnaryDTypeKernels(dtype.Float16)
}

func registerUnaryBFloat16Kernels() {
	registerUnaryDTypeKernels(dtype.BFloat16)
}

func registerUnaryDTypeKernels(storageDType dtype.DType) {
	registerUnaryKernel("relu", storageDType, runUnaryElementwise(metalUnaryFloat32Relu))
	registerUnaryKernel("abs", storageDType, runUnaryElementwise(metalUnaryFloat32Abs))
	registerUnaryKernel("neg", storageDType, runUnaryElementwise(metalUnaryFloat32Neg))
	registerUnaryKernel("square", storageDType, runUnaryElementwise(metalUnaryFloat32Square))
	registerUnaryKernel("recip", storageDType, runUnaryElementwise(metalUnaryFloat32Recip))
	registerUnaryKernel("sqrt", storageDType, runUnaryElementwise(metalUnaryFloat32Sqrt))
	registerUnaryKernel("sign", storageDType, runUnaryElementwise(metalUnaryFloat32Sign))
}

func registerUnaryKernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func (backend *Backend) unaryFloat32(
	ctx context.Context,
	operation metalUnaryFloat32Operation,
	input tensor.Tensor,
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

	if input.DType() != dtype.Float32 {
		return nil, tensor.ErrDTypeMismatch
	}

	out, err := backend.bridge.empty(input.Shape(), dtype.Float32)
	if err != nil {
		return nil, err
	}

	if err := runMetalUnaryFloat32(operation, input, out); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func runUnaryFloat32(operation metalUnaryFloat32Operation) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		return runMetalUnaryFloat32(operation, args[0], args[1])
	}
}

func runUnaryElementwise(operation metalUnaryFloat32Operation) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		return runMetalUnaryElementwise(operation, args[0], args[1])
	}
}
