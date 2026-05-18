package kernels

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Backward kernels paired with the forward implementations in this
package. Per AGENTS.md §1 every backward kernel ships scalar Go plus
the four host SIMD ISAs and the device backends; this file establishes
the scalar Go reference paths and the registration shape.

The forward kernels in add.go / matmul.go / elementwise.go / etc.
register their backward closures as `SimpleGradFn.BackFn` when they
are called with autograd-enabled inputs. The reference closures live
here so they can be reused across forward signatures.

Per the spray-and-pray contract, the bodies are correct in
elementwise arithmetic; mixed-dtype accumulation conventions match
the forward kernel for the same op.
*/

/*
AddBackward returns gradients that match the upstream gradient for
each input. d(a+b)/da = 1, d(a+b)/db = 1, so both inputs receive the
upstream gradient unchanged.
*/
func AddBackward(ctx context.Context, upstream tensor.Tensor) ([]tensor.Tensor, error) {
	leftGrad, err := tensor.Contiguous(upstream)

	if err != nil {
		return nil, err
	}

	rightGrad, err := tensor.Contiguous(upstream)

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{leftGrad, rightGrad}, nil
}

/*
SubBackward returns +upstream for the left input and -upstream for
the right. The negation is done via a fresh allocation, not in
place.
*/
func SubBackward(ctx context.Context, upstream tensor.Tensor) ([]tensor.Tensor, error) {
	leftGrad, err := tensor.Contiguous(upstream)

	if err != nil {
		return nil, err
	}

	upstreamView, err := upstream.Float32Native()

	if err != nil {
		return nil, err
	}

	rightShape := upstream.Shape()
	rightGrad, err := tensor.NewZeroed(rightShape, dtype.Float32)

	if err != nil {
		return nil, err
	}

	rightView, err := rightGrad.Float32Native()

	if err != nil {
		return nil, err
	}

	for index, value := range upstreamView {
		rightView[index] = -value
	}

	return []tensor.Tensor{leftGrad, rightGrad}, nil
}

/*
MulBackward implements d(a*b)/da = b and d(a*b)/db = a. Requires the
captured forward inputs (left, right) which the kernel closure
provides via SimpleGradFn.InputList.
*/
func MulBackward(left, right tensor.Tensor) func(context.Context, tensor.Tensor) ([]tensor.Tensor, error) {
	return func(ctx context.Context, upstream tensor.Tensor) ([]tensor.Tensor, error) {
		upstreamView, err := upstream.Float32Native()

		if err != nil {
			return nil, err
		}

		leftView, err := left.Float32Native()

		if err != nil {
			return nil, err
		}

		rightView, err := right.Float32Native()

		if err != nil {
			return nil, err
		}

		shape := upstream.Shape()

		leftGrad, _ := tensor.NewZeroed(shape, dtype.Float32)
		rightGrad, _ := tensor.NewZeroed(shape, dtype.Float32)

		leftGradView, _ := leftGrad.Float32Native()
		rightGradView, _ := rightGrad.Float32Native()

		for index, upValue := range upstreamView {
			leftGradView[index] = upValue * rightView[index]
			rightGradView[index] = upValue * leftView[index]
		}

		return []tensor.Tensor{leftGrad, rightGrad}, nil
	}
}

/*
MatMulBackward implements the gradient of C = A @ B:

	dC/dA = upstream @ B^T
	dC/dB = A^T @ upstream

The closure captures the forward A and B; upstream has the shape of
C. Phase 11 expansion routes through dedicated matmul-transpose
kernels for performance; this reference body uses two explicit
multiplications with on-the-fly transposition.
*/
func MatMulBackward(left, right tensor.Tensor) func(context.Context, tensor.Tensor) ([]tensor.Tensor, error) {
	return func(ctx context.Context, upstream tensor.Tensor) ([]tensor.Tensor, error) {
		leftDims := left.Shape().Dims()
		rightDims := right.Shape().Dims()

		if len(leftDims) != 2 || len(rightDims) != 2 {
			return nil, tensor.ErrShapeMismatch
		}

		rows := leftDims[0]
		inner := leftDims[1]
		cols := rightDims[1]

		upstreamView, err := upstream.Float32Native()

		if err != nil {
			return nil, err
		}

		leftView, err := left.Float32Native()

		if err != nil {
			return nil, err
		}

		rightView, err := right.Float32Native()

		if err != nil {
			return nil, err
		}

		leftGradShape, _ := tensor.NewShape([]int{rows, inner})
		rightGradShape, _ := tensor.NewShape([]int{inner, cols})

		leftGrad, _ := tensor.NewZeroed(leftGradShape, dtype.Float32)
		rightGrad, _ := tensor.NewZeroed(rightGradShape, dtype.Float32)

		leftGradView, _ := leftGrad.Float32Native()
		rightGradView, _ := rightGrad.Float32Native()

		// dA = upstream @ B^T
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			for innerIndex := 0; innerIndex < inner; innerIndex++ {
				var sum float32

				for colIndex := 0; colIndex < cols; colIndex++ {
					sum += upstreamView[rowIndex*cols+colIndex] *
						rightView[innerIndex*cols+colIndex]
				}

				leftGradView[rowIndex*inner+innerIndex] = sum
			}
		}

		// dB = A^T @ upstream
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			for colIndex := 0; colIndex < cols; colIndex++ {
				var sum float32

				for rowIndex := 0; rowIndex < rows; rowIndex++ {
					sum += leftView[rowIndex*inner+innerIndex] *
						upstreamView[rowIndex*cols+colIndex]
				}

				rightGradView[innerIndex*cols+colIndex] = sum
			}
		}

		return []tensor.Tensor{leftGrad, rightGrad}, nil
	}
}

/*
ReLUBackward implements d(relu)/dx = 1 if x > 0 else 0. Closure
captures the forward input.
*/
func ReLUBackward(input tensor.Tensor) func(context.Context, tensor.Tensor) ([]tensor.Tensor, error) {
	return func(ctx context.Context, upstream tensor.Tensor) ([]tensor.Tensor, error) {
		upstreamView, err := upstream.Float32Native()

		if err != nil {
			return nil, err
		}

		inputView, err := input.Float32Native()

		if err != nil {
			return nil, err
		}

		shape := upstream.Shape()

		gradient, _ := tensor.NewZeroed(shape, dtype.Float32)
		gradView, _ := gradient.Float32Native()

		for index, value := range upstreamView {
			gradView[index] = 0

			if inputView[index] > 0 {
				gradView[index] = value
			}
		}

		return []tensor.Tensor{gradient}, nil
	}
}
