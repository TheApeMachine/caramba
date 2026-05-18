package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Sparse matmul for CSR-format tensors. The host reference does the
straight-forward row-by-row traversal; vendor primitives (cuSPARSE,
MPS-Graph sparse) handle device fast paths in Phase 9 expansion.

Args: (sparseLeft [CSR, Float32], denseRight [Dense, Float32],
       output [Dense, Float32]).

Sparse left × dense right → dense output is the dominant pattern for
inference (sparse weights, dense activations); the symmetric cases
(dense × sparse, sparse × sparse) land in later sessions.
*/

func init() {
	Default.Register(Kernel{
		Name: "sparse_csr_matmul",
		Signature: Signature{
			Layout: tensor.LayoutSparseCSR,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSparseCSRMatMulDefault,
	})
}

func runSparseCSRMatMulDefault(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return SparseCSRMatMulFloat32(args[0], args[1], args[2])
}

/*
SparseCSRMatMulFloat32 computes output[i,j] = sum_k sparse[i,k] *
denseRight[k,j]. The sparse left must be a SparseTensor with CSR
layout; denseRight and output must be plain dense float32 tensors
in row-major order.
*/
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

	sparseCSRMatMulFloat32Native(
		outView, valuesView, rightView,
		rowPtr, colIdx,
		rows, cols,
	)

	return nil
}

func sparseCSRMatMulFloat32Scalar(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	for index := range outView {
		outView[index] = 0
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowStart := int(rowPtr[rowIndex])
		rowEnd := int(rowPtr[rowIndex+1])

		for nzIndex := rowStart; nzIndex < rowEnd; nzIndex++ {
			colInLeft := int(colIdx[nzIndex])
			value := valuesView[nzIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				outView[rowIndex*cols+colIndex] +=
					value * rightView[colInLeft*cols+colIndex]
			}
		}
	}
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
