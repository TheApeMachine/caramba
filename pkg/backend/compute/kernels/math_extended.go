package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Additional math kernels found in the original substrate's math
package that aren't covered by the elementwise drivers:

  - inv_sqrt_dim_scale: y = x / sqrt(dim). The attention scaling
    primitive that produces 1 / √d_k for a head dimension.
  - logsumexp:           y = log(Σ exp(x_i - max(x))) + max(x).
                         Numerically stable log-sum-exp reduction.
  - matmul_add:          fused (A @ B) + bias.
  - outer:               outer product: y[i,j] = a[i] × b[j].
*/

func init() {
	Default.Register(Kernel{
		Name: "inv_sqrt_dim_scale",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runInvSqrtDimScale,
	})

	Default.Register(Kernel{
		Name: "logsumexp",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLogSumExp,
	})

	Default.Register(Kernel{
		Name: "matmul_add",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMatMulAdd,
	})

	Default.Register(Kernel{
		Name: "outer",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runOuter,
	})
}

/*
runInvSqrtDimScale reads input [N] and a scalar dim [1] of int32 and
writes out[i] = input[i] / sqrt(dim).
*/
func runInvSqrtDimScale(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	dim, _ := args[1].Int32Native()
	out, _ := args[2].Float32Native()

	if len(out) != len(input) || len(dim) < 1 {
		return tensor.ErrShapeMismatch
	}

	if dim[0] <= 0 {
		return tensor.ErrShapeMismatch
	}

	scale := float32(1.0 / math.Sqrt(float64(dim[0])))

	for index, value := range input {
		out[index] = value * scale
	}

	return nil
}

/*
runLogSumExp computes the row-wise log-sum-exp across the last
dimension. Output preserves all leading dims and reduces to a scalar
along the trailing axis.
*/
func runLogSumExp(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	out, _ := args[1].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) == 0 {
		return tensor.ErrShapeMismatch
	}

	lastDim := dims[len(dims)-1]

	if len(input)%lastDim != 0 || len(out) != len(input)/lastDim {
		return tensor.ErrShapeMismatch
	}

	rows := len(input) / lastDim

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		maximum := row[0]

		for _, candidate := range row[1:] {
			if candidate > maximum {
				maximum = candidate
			}
		}

		var accumulator float64

		for _, candidate := range row {
			accumulator += math.Exp(float64(candidate - maximum))
		}

		out[rowIndex] = maximum + float32(math.Log(accumulator))
	}

	return nil
}

/*
runMatMulAdd computes (A @ B) + bias for [M, K] × [K, N] + [N].
Output shape is [M, N].
*/
func runMatMulAdd(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	aView, _ := args[0].Float32Native()
	bView, _ := args[1].Float32Native()
	biasView, _ := args[2].Float32Native()
	outView, _ := args[3].Float32Native()

	aDims := args[0].Shape().Dims()
	bDims := args[1].Shape().Dims()
	biasDims := args[2].Shape().Dims()

	if len(aDims) != 2 || len(bDims) != 2 || len(biasDims) != 1 ||
		aDims[1] != bDims[0] || biasDims[0] != bDims[1] {
		return tensor.ErrShapeMismatch
	}

	rows := aDims[0]
	inner := aDims[1]
	cols := bDims[1]

	if len(outView) != rows*cols {
		return tensor.ErrShapeMismatch
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for colIndex := 0; colIndex < cols; colIndex++ {
			sum := biasView[colIndex]

			for innerIndex := 0; innerIndex < inner; innerIndex++ {
				sum += aView[rowIndex*inner+innerIndex] *
					bView[innerIndex*cols+colIndex]
			}

			outView[rowIndex*cols+colIndex] = sum
		}
	}

	return nil
}

/*
runOuter computes the outer product y[i, j] = a[i] × b[j]. Output
shape is [len(a), len(b)].
*/
func runOuter(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	aView, _ := args[0].Float32Native()
	bView, _ := args[1].Float32Native()
	outView, _ := args[2].Float32Native()

	if len(outView) != len(aView)*len(bView) {
		return tensor.ErrShapeMismatch
	}

	for aIndex, aValue := range aView {
		for bIndex, bValue := range bView {
			outView[aIndex*len(bView)+bIndex] = aValue * bValue
		}
	}

	return nil
}
