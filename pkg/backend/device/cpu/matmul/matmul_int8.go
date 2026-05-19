package matmul

import (
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dot"
)

/*
INT8 matmul with int32 accumulation. Inputs are int8 in standard
row-major layout; the output is int32 (per-tensor scale and zero-point
are applied via a separate dequant step in the inference graph).

Pipeline:
  - Transpose RHS into a packed buffer so each "column" of the original
    B becomes a contiguous row, allowing SDOT to load 16 contiguous
    int8 values per call.
  - For each (output row, output col) pair, compute the int8 dot via
    the NEON SDOT-based kernel.

Math contract: out[i,j] = sum_k(int32(lhs[i,k]) * int32(rhs[k,j])),
exact integer arithmetic with int32 accumulator (no saturation; the
domain has plenty of headroom for typical model sizes since |a*b| ≤
128*128 = 16384 and 65536 accumulations stay within int32 range).
*/

var int8MatmulPool = sync.Pool{
	New: func() any {
		buf := make([]int8, 0, 4096)
		return &buf
	},
}

func borrowInt8Buffer(n int) []int8 {
	bufPtr := int8MatmulPool.Get().(*[]int8)
	buf := *bufPtr

	if cap(buf) < n {
		buf = make([]int8, n)
	} else {
		buf = buf[:n]
	}

	return buf
}

func releaseInt8Buffer(buf []int8) {
	buf = buf[:0]
	int8MatmulPool.Put(&buf)
}

func RunMatMulInt8(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	lhs, rhs, out := args[0], args[1], args[2]

	rows, inner, cols, err := matmulDims(lhs, rhs, out)

	if err != nil {
		return err
	}

	leftView, err := lhs.Int8Native()

	if err != nil {
		return err
	}

	rightView, err := rhs.Int8Native()

	if err != nil {
		return err
	}

	outView, err := out.Int32Native()

	if err != nil {
		return err
	}

	// Pack RHS in column-major: each "column" of B becomes a contiguous
	// row of length K. Output dimensions stay (rows, cols).
	packed := borrowInt8Buffer(inner * cols)
	defer releaseInt8Buffer(packed)

	for col := range cols {
		for k := range inner {
			packed[col*inner+k] = rightView[k*cols+col]
		}
	}

	// For each (row, col), dot the row of A against the packed column of B.
	for row := range rows {
		rowSlice := leftView[row*inner : row*inner+inner]
		outRow := outView[row*cols : row*cols+cols]

		for col := range cols {
			colSlice := packed[col*inner : col*inner+inner]
			outRow[col] = dot.DotInt8Native(rowSlice, colSlice)
		}
	}

	return nil
}
