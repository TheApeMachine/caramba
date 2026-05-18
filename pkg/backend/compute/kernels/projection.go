package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Projection kernels — the canonical dense linear layers used by
transformer blocks.

  - linear: y = x @ W^T + b (PyTorch convention).
  - fused_qkv: computes Q, K, V projections in a single pass against
    a fused [3 × outDim, inDim] weight matrix.
*/

func init() {
	Default.Register(Kernel{
		Name: "linear",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLinear,
	})

	Default.Register(Kernel{
		Name: "fused_qkv",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runFusedQKV,
	})
}

/*
runLinear computes y = x @ W^T + b where x is [batch, inDim],
W is [outDim, inDim], b is [outDim], y is [batch, outDim].
*/
func runLinear(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	xView, _ := args[0].Float32Native()
	wView, _ := args[1].Float32Native()
	bView, _ := args[2].Float32Native()
	yView, _ := args[3].Float32Native()

	xDims := args[0].Shape().Dims()
	wDims := args[1].Shape().Dims()
	bDims := args[2].Shape().Dims()
	yDims := args[3].Shape().Dims()

	if len(xDims) != 2 || len(wDims) != 2 ||
		len(bDims) != 1 || len(yDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	batch := xDims[0]
	inDim := xDims[1]
	outDim := wDims[0]

	if wDims[1] != inDim || bDims[0] != outDim ||
		yDims[0] != batch || yDims[1] != outDim {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := bView[outIndex]

			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += xView[batchIndex*inDim+inIndex] *
					wView[outIndex*inDim+inIndex]
			}

			yView[batchIndex*outDim+outIndex] = sum
		}
	}

	return nil
}

/*
runFusedQKV computes Q, K, V in a single matmul against a fused
[3 × headDim × numHeads, inDim] weight matrix. Args:
(input, fusedWeight, bias) → (queryOut, keyOut, valueOut). The
bias is also fused: [3 × headDim × numHeads].

Output split: rows 0..outDim → Q, outDim..2×outDim → K,
2×outDim..3×outDim → V (where outDim = headDim × numHeads).
*/
func runFusedQKV(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	xView, _ := args[0].Float32Native()
	wView, _ := args[1].Float32Native()
	bView, _ := args[2].Float32Native()
	qView, _ := args[3].Float32Native()
	kView, _ := args[4].Float32Native()
	vView, _ := args[5].Float32Native()

	xDims := args[0].Shape().Dims()
	wDims := args[1].Shape().Dims()

	if len(xDims) != 2 || len(wDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	batch := xDims[0]
	inDim := xDims[1]
	fusedOut := wDims[0]

	if fusedOut%3 != 0 || wDims[1] != inDim || len(bView) != fusedOut {
		return tensor.ErrShapeMismatch
	}

	outDim := fusedOut / 3

	if len(qView) != batch*outDim ||
		len(kView) != batch*outDim ||
		len(vView) != batch*outDim {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := range batch {
		for outIndex := range outDim {
			qSum := bView[outIndex]
			kSum := bView[outDim+outIndex]
			vSum := bView[2*outDim+outIndex]

			for inIndex := range inDim {
				inputValue := xView[batchIndex*inDim+inIndex]
				qSum += inputValue * wView[outIndex*inDim+inIndex]
				kSum += inputValue * wView[(outDim+outIndex)*inDim+inIndex]
				vSum += inputValue * wView[(2*outDim+outIndex)*inDim+inIndex]
			}

			qView[batchIndex*outDim+outIndex] = qSum
			kView[batchIndex*outDim+outIndex] = kSum
			vView[batchIndex*outDim+outIndex] = vSum
		}
	}

	return nil
}
