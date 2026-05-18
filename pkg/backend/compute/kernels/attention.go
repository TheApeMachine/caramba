package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
ScaledDotProductAttention is the host reference for the canonical
attention kernel: softmax(Q @ K^T / sqrt(d_k)) @ V.

Inputs (in args order): query, key, value, output. Query and key
share dtype and inner dim; value's outer dims match query, its inner
dim becomes the output's inner dim.

Per the spray-and-pray contract this is the scalar Go reference. The
flash-attention block-tiled variants, alibi/rope hooks, and sliding-
window masking land in Phase 8 expansion sessions with their own
Kernel registrations.

Tensor shapes (no batch dim — Phase 8 expansion adds batched
attention by extending the dispatch table):
  - query  [seqQ, depth]
  - key    [seqK, depth]
  - value  [seqK, valueDim]
  - output [seqQ, valueDim]
*/

func init() {
	Default.Register(Kernel{
		Name: "attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAttentionFloat32,
	})
}

func runAttentionFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]

	queryDims := query.Shape().Dims()
	keyDims := key.Shape().Dims()
	valueDims := value.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 ||
		len(valueDims) != 2 || len(outDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := queryDims[0]
	depth := queryDims[1]
	seqK := keyDims[0]
	valueDim := valueDims[1]

	if keyDims[1] != depth || valueDims[0] != seqK ||
		outDims[0] != seqQ || outDims[1] != valueDim {
		return tensor.ErrShapeMismatch
	}

	queryView, err := query.Float32Native()

	if err != nil {
		return err
	}

	keyView, err := key.Float32Native()

	if err != nil {
		return err
	}

	valueView, err := value.Float32Native()

	if err != nil {
		return err
	}

	outView, err := out.Float32Native()

	if err != nil {
		return err
	}

	scale := float32(1.0 / math.Sqrt(float64(depth)))
	scores := make([]float32, seqQ*seqK)

	// scores = Q @ K^T * scale.
	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for keyIndex := 0; keyIndex < seqK; keyIndex++ {
			var dot float32

			for depthIndex := 0; depthIndex < depth; depthIndex++ {
				dot += queryView[rowIndex*depth+depthIndex] *
					keyView[keyIndex*depth+depthIndex]
			}

			scores[rowIndex*seqK+keyIndex] = dot * scale
		}
	}

	// Stable row-wise softmax.
	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		row := scores[rowIndex*seqK : (rowIndex+1)*seqK]
		maximum := row[0]

		for _, candidate := range row[1:] {
			if candidate > maximum {
				maximum = candidate
			}
		}

		var sum float32

		for index, candidate := range row {
			shifted := float32(math.Exp(float64(candidate - maximum)))
			row[index] = shifted
			sum += shifted
		}

		if sum == 0 {
			continue
		}

		for index := range row {
			row[index] /= sum
		}
	}

	// output = softmax(scores) @ V.
	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			var weighted float32

			for keyIndex := 0; keyIndex < seqK; keyIndex++ {
				weighted += scores[rowIndex*seqK+keyIndex] *
					valueView[keyIndex*valueDim+dimIndex]
			}

			outView[rowIndex*valueDim+dimIndex] = weighted
		}
	}

	return nil
}
