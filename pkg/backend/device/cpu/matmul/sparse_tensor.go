package matmul

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func RunSparseCSRMatMulDefault(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return SparseCSRMatMulFloat32(args[0], args[1], args[2])
}

func SparseCSRMatMulFloat32(
	sparseLeft, denseRight, output tensor.Tensor,
) error {
	sparse, ok := sparseLeft.(tensor.SparseTensor)

	if !ok {
		return tensor.ErrLayoutUnsupported
	}

	if sparse.Layout() != tensor.LayoutSparseCSR {
		return tensor.ErrLayoutUnsupported
	}

	leftDims := sparse.Shape().Dims()
	rightDims := denseRight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 || len(outDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	rows := leftDims[0]
	innerLeft := leftDims[1]
	innerRight := rightDims[0]
	cols := rightDims[1]

	if innerLeft != innerRight ||
		outDims[0] != rows || outDims[1] != cols {
		return tensor.ErrShapeMismatch
	}

	values, err := sparse.Values()

	if err != nil {
		return err
	}

	valuesView, err := values.Float32Native()

	if err != nil {
		return err
	}

	indices, err := sparse.Indices()

	if err != nil {
		return err
	}

	rowPtr, colIdx, err := extractCSRIndices(indices)

	if err != nil {
		return err
	}

	rightView, err := denseRight.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	SparseCSRMatMulFloat32Native(
		outView, valuesView, rightView,
		rowPtr, colIdx,
		rows, cols,
	)

	return nil
}

func extractCSRIndices(indices []tensor.SparseIndex) ([]int32, []int32, error) {
	var rowPtr, colIdx []int32

	for _, index := range indices {
		view, err := index.Data.Int32Native()

		if err != nil {
			return nil, nil, err
		}

		switch index.Name {
		case "row_ptr":
			rowPtr = view
		case "col_idx":
			colIdx = view
		}
	}

	if rowPtr == nil || colIdx == nil {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return rowPtr, colIdx, nil
}
