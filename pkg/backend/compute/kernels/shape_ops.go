package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Shape-manipulation kernels: gather, scatter, concat, split, expand,
transpose, masked_fill, where. These are the data-movement primitives
that live above the contiguous-storage contract. Strided rearrangements
materialize via these kernels.
*/

func init() {
	Default.Register(Kernel{
		Name: "gather",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runGatherFloat32Int32,
	})

	Default.Register(Kernel{
		Name: "scatter",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runScatterFloat32Int32,
	})

	Default.Register(Kernel{
		Name: "where",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Bool, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runWhereFloat32,
	})

	Default.Register(Kernel{
		Name: "masked_fill",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Bool, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMaskedFillFloat32,
	})

	Default.Register(Kernel{
		Name: "transpose2d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runTranspose2DFloat32,
	})

	Default.Register(Kernel{
		Name: "concat",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runConcatFloat32,
	})
}

/*
Gather selects rows from a [N, D] source by indices [M] producing a
[M, D] output.
*/
func runGatherFloat32Int32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	source, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	sourceDims := args[0].Shape().Dims()

	if len(sourceDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	innerDim := sourceDims[1]

	if len(out) != len(indices)*innerDim {
		return tensor.ErrShapeMismatch
	}

	for resultIndex, sourceRow := range indices {
		if int(sourceRow) < 0 || int(sourceRow) >= sourceDims[0] {
			return tensor.ErrShapeMismatch
		}

		copy(
			out[resultIndex*innerDim:(resultIndex+1)*innerDim],
			source[int(sourceRow)*innerDim:(int(sourceRow)+1)*innerDim],
		)
	}

	return nil
}

/*
Scatter writes rows from updates [M, D] to target [N, D] at indices
[M]. The args order is (target, indices, updates, output).
*/
func runScatterFloat32Int32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	target, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	updates, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[3].Float32Native()

	if err != nil {
		return err
	}

	targetDims := args[0].Shape().Dims()

	if len(targetDims) != 2 || len(out) != len(target) {
		return tensor.ErrShapeMismatch
	}

	innerDim := targetDims[1]

	if len(updates) != len(indices)*innerDim {
		return tensor.ErrShapeMismatch
	}

	copy(out, target)

	for updateIndex, targetRow := range indices {
		if int(targetRow) < 0 || int(targetRow) >= targetDims[0] {
			return tensor.ErrShapeMismatch
		}

		copy(
			out[int(targetRow)*innerDim:(int(targetRow)+1)*innerDim],
			updates[updateIndex*innerDim:(updateIndex+1)*innerDim],
		)
	}

	return nil
}

/*
Where selects entries from positive/negative based on a boolean mask:
out[i] = mask[i] ? positive[i] : negative[i].
*/
func runWhereFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	mask, err := args[0].BoolNative()

	if err != nil {
		return err
	}

	positive, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	negative, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[3].Float32Native()

	if err != nil {
		return err
	}

	if len(positive) != len(negative) ||
		len(out) != len(positive) ||
		mask.Len() != len(positive) {
		return tensor.ErrShapeMismatch
	}

	for index := range out {
		if mask.Get(index) {
			out[index] = positive[index]
			continue
		}

		out[index] = negative[index]
	}

	return nil
}

/*
MaskedFill replaces input entries where mask is true with the value
read from the scalar tensor (length-1 float32). Output preserves
input dtype/shape.
*/
func runMaskedFillFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	mask, err := args[1].BoolNative()

	if err != nil {
		return err
	}

	scalar, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[3].Float32Native()

	if err != nil {
		return err
	}

	if len(out) != len(input) || mask.Len() != len(input) || len(scalar) < 1 {
		return tensor.ErrShapeMismatch
	}

	fillValue := scalar[0]

	for index := range out {
		out[index] = input[index]

		if mask.Get(index) {
			out[index] = fillValue
		}
	}

	return nil
}

/*
Transpose2D swaps the two axes of a 2-D contiguous tensor.
*/
func runTranspose2DFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	inDims := args[0].Shape().Dims()
	outDims := args[1].Shape().Dims()

	if len(inDims) != 2 || len(outDims) != 2 ||
		inDims[0] != outDims[1] || inDims[1] != outDims[0] {
		return tensor.ErrShapeMismatch
	}

	rows := inDims[0]
	cols := inDims[1]

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			out[col*rows+row] = input[row*cols+col]
		}
	}

	return nil
}

/*
Concat concatenates two same-rank tensors along axis 0. Phase 8
expansion adds the general N-axis form; the host reference here
covers the most common case (concat-along-batch / concat-along-seq).
*/
func runConcatFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	if len(out) != len(left)+len(right) {
		return tensor.ErrShapeMismatch
	}

	copy(out, left)
	copy(out[len(left):], right)

	return nil
}
