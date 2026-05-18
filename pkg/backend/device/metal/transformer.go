package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalTransformerDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalTransformerDTypes {
		registerMetalTransformerKernels(storageDType)
	}
}

func registerMetalTransformerKernels(storageDType dtype.DType) {
	registerMetalAttentionKernel(storageDType)
	registerMetalEmbeddingLookupKernel(storageDType)
	registerMetalEmbeddingBagKernel(storageDType)
	registerMetalApplyMaskKernel(storageDType)
	registerMetalCausalMaskKernel(storageDType)
	registerMetalALiBiBiasKernel(storageDType)
}

func registerMetalAttentionKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "attention",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalAttentionKernel,
	})
}

func registerMetalEmbeddingLookupKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "embedding_lookup",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalEmbeddingLookupKernel,
	})
}

func registerMetalEmbeddingBagKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "embedding_bag",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, dtype.Int32, dtype.Int32},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalEmbeddingBagKernel,
	})
}

func registerMetalApplyMaskKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "apply_mask",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalApplyMaskKernel,
	})
}

func registerMetalCausalMaskKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "causal_mask",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalCausalMaskKernel,
	})
}

func registerMetalALiBiBiasKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "alibi_bias",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalALiBiBiasKernel,
	})
}

func runMetalAttentionKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalAttention(args[0], args[1], args[2], args[3])
}

func runMetalEmbeddingLookupKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalEmbeddingLookup(args[0], args[1], args[2])
}

func runMetalEmbeddingBagKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalEmbeddingBag(args[0], args[1], args[2], args[3])
}

func runMetalApplyMaskKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalApplyMask(args[0], args[1], args[2])
}

func runMetalCausalMaskKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalCausalMask(args[0], args[1])
}

func runMetalALiBiBiasKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalALiBiBias(args[0], args[1], args[2])
}
