package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalShapeDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalShapeDTypes {
		registerMetalShapeKernels(storageDType)
	}
}

func registerMetalShapeKernels(storageDType dtype.DType) {
	registerMetalGatherKernel(storageDType)
	registerMetalScatterKernel(storageDType)
	registerMetalWhereKernel(storageDType)
	registerMetalMaskedFillKernel(storageDType)
	registerMetalUnaryShapeKernel("last_token", storageDType, runMetalLastToken)
	registerMetalUnaryShapeKernel("merge_heads", storageDType, runMetalMergeHeads)
	registerMetalUnaryShapeKernel("split_heads", storageDType, runMetalSplitHeads)
	registerMetalUnaryShapeKernel("reshape", storageDType, runMetalReshape)
	registerMetalSliceKernel(storageDType)
	registerMetalTransposeKernel(storageDType)
	registerMetalUnaryShapeKernel("transpose2d", storageDType, runMetalTranspose2D)
	registerMetalUnaryShapeKernel("upsample_nearest2d", storageDType, runMetalUpsampleNearest2D)
	registerMetalBinaryShapeKernel("concat", storageDType, runMetalConcat)
	registerMetalSplit2Kernel(storageDType)
	registerMetalViewAsHeadsKernel(storageDType)
}

func registerMetalGatherKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "gather",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runGatherShape(runMetalGather),
	})
}

func registerMetalScatterKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "scatter",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runScatterShape(runMetalScatter),
	})
}

func registerMetalWhereKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "where",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Bool, storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runWhereShape(runMetalWhere),
	})
}

func registerMetalMaskedFillKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "masked_fill",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Bool, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMaskedFillShape(runMetalMaskedFill),
	})
}

func registerMetalTransposeKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "transpose",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runViewAsHeadsShape(runMetalTranspose),
	})
}

func registerMetalUnaryShapeKernel(
	name string,
	storageDType dtype.DType,
	run func(tensor.Tensor, tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runUnaryShape(run),
	})
}

func registerMetalBinaryShapeKernel(
	name string,
	storageDType dtype.DType,
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runBinaryShape(run),
	})
}

func registerMetalSplit2Kernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "split2",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType, storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runSplit2Shape(runMetalSplit2),
	})
}

func registerMetalSliceKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "slice",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				dtype.Int32,
				dtype.Int32,
				dtype.Int32,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runSliceShape(runMetalSlice),
	})
}

func registerMetalViewAsHeadsKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "view_as_heads",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runViewAsHeadsShape(runMetalViewAsHeads),
	})
}

func runUnaryShape(
	run func(tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1])
	}
}

func runBinaryShape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2])
	}
}

func runSplit2Shape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2])
	}
}

func runViewAsHeadsShape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2])
	}
}

func runSliceShape(
	run func(
		tensor.Tensor,
		tensor.Tensor,
		tensor.Tensor,
		tensor.Tensor,
		tensor.Tensor,
	) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 5 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2], args[3], args[4])
	}
}

func runGatherShape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2])
	}
}

func runScatterShape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 4 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2], args[3])
	}
}

func runWhereShape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 4 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2], args[3])
	}
}

func runMaskedFillShape(
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) func(...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 4 {
			return tensor.ErrShapeMismatch
		}

		return run(args[0], args[1], args[2], args[3])
	}
}
