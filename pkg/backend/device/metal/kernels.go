package metal

import (
	"context"
	"slices"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	kernels.Default.Register(kernels.Kernel{
		Name: "add",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runAddFloat32,
	})
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

	if err := runMetalAddFloat32(left, right, out); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func runAddFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalAddFloat32(args[0], args[1], args[2])
}
