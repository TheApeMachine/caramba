package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Additional shape-manipulation kernels found in the canonical
transformer pipeline: last_token, merge_heads, split_heads, slice,
reshape (metadata-only), split.

  - last_token: extracts the final timestep from a [batch, seq, hidden]
    tensor.
  - merge_heads: collapses [batch, seq, heads, headDim] →
    [batch, seq, heads × headDim].
  - split_heads: inverse of merge_heads.
  - slice: copies a contiguous range out of the source.
  - split: splits a tensor into two halves along axis 0.
*/

func init() {
	Default.Register(Kernel{
		Name: "last_token",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLastToken,
	})

	Default.Register(Kernel{
		Name: "merge_heads",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMergeHeads,
	})

	Default.Register(Kernel{
		Name: "split_heads",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSplitHeads,
	})

	Default.Register(Kernel{
		Name: "split2",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32, dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSplit2,
	})
}

func runLastToken(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	outView, _ := args[1].Float32Native()

	inDims := args[0].Shape().Dims()

	if len(inDims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	seq := inDims[1]
	hidden := inDims[2]

	if len(outView) != batch*hidden {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		lastTokenOffset := (batchIndex*seq + seq - 1) * hidden
		outOffset := batchIndex * hidden

		copy(outView[outOffset:outOffset+hidden], inView[lastTokenOffset:lastTokenOffset+hidden])
	}

	return nil
}

func runMergeHeads(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	outView, _ := args[1].Float32Native()

	inDims := args[0].Shape().Dims()

	if len(inDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	seq := inDims[1]
	heads := inDims[2]
	headDim := inDims[3]

	if len(outView) != batch*seq*heads*headDim {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for seqIndex := 0; seqIndex < seq; seqIndex++ {
			for headIndex := 0; headIndex < heads; headIndex++ {
				inOffset := ((batchIndex*seq+seqIndex)*heads + headIndex) * headDim
				outOffset := ((batchIndex*seq + seqIndex) * heads * headDim) + headIndex*headDim

				copy(outView[outOffset:outOffset+headDim], inView[inOffset:inOffset+headDim])
			}
		}
	}

	return nil
}

func runSplitHeads(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	outView, _ := args[1].Float32Native()

	outDims := args[1].Shape().Dims()

	if len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := outDims[0]
	seq := outDims[1]
	heads := outDims[2]
	headDim := outDims[3]

	if len(inView) != batch*seq*heads*headDim {
		return tensor.ErrShapeMismatch
	}

	copy(outView, inView)
	return nil
}

func runSplit2(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	leftView, _ := args[1].Float32Native()
	rightView, _ := args[2].Float32Native()

	if len(inView) != len(leftView)+len(rightView) {
		return tensor.ErrShapeMismatch
	}

	copy(leftView, inView[:len(leftView)])
	copy(rightView, inView[len(leftView):])

	return nil
}
