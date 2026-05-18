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

Body decomposed into computeAttentionScores, applySoftmax, and
computeWeightedOutput so runAttentionFloat32 stays small and each
phase is independently testable.

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

	queryView, keyView, valueView, outView, seqQ, seqK, depth, valueDim, err := attentionViews(query, key, value, out)

	if err != nil {
		return err
	}

	scale := float32(1.0 / math.Sqrt(float64(depth)))
	scores := computeAttentionScores(queryView, keyView, seqQ, seqK, depth, scale)
	applySoftmax(scores, seqQ, seqK)
	computeWeightedOutput(scores, valueView, outView, seqQ, seqK, valueDim)

	return nil
}

func attentionViews(
	query, key, value, out tensor.Tensor,
) (qv, kv, vv, ov []float32, seqQ, seqK, depth, valueDim int, err error) {
	queryDims := query.Shape().Dims()
	keyDims := key.Shape().Dims()
	valueDims := value.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 ||
		len(valueDims) != 2 || len(outDims) != 2 {
		return nil, nil, nil, nil, 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	seqQ = queryDims[0]
	depth = queryDims[1]
	seqK = keyDims[0]
	valueDim = valueDims[1]

	if keyDims[1] != depth || valueDims[0] != seqK ||
		outDims[0] != seqQ || outDims[1] != valueDim {
		return nil, nil, nil, nil, 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	qv, err = query.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, 0, err
	}

	kv, err = key.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, 0, err
	}

	vv, err = value.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, 0, err
	}

	ov, err = out.Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, 0, err
	}

	return qv, kv, vv, ov, seqQ, seqK, depth, valueDim, nil
}

/*
computeAttentionScores returns Q @ K^T × scale. The result is a
[seqQ, seqK] row-major slice.
*/
func computeAttentionScores(
	queryView, keyView []float32,
	seqQ, seqK, depth int,
	scale float32,
) []float32 {
	scores := make([]float32, seqQ*seqK)

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

	return scores
}

/*
applySoftmax performs row-wise stable softmax in place on a
[seqQ, seqK] score matrix.
*/
func applySoftmax(scores []float32, seqQ, seqK int) {
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

		// Degenerate but expected case: every score in this row was
		// extremely negative (effectively -Inf), so every shifted
		// exp underflowed to zero. The row stays all-zeros, which
		// produces a zero attention output for this query position.
		// Callers (causal masking with all preceding positions
		// masked, padded-batch slots, fully dropped heads)
		// intentionally produce this state, so we continue rather
		// than raise an error.
		if sum == 0 {
			continue
		}

		for index := range row {
			row[index] /= sum
		}
	}
}

/*
computeWeightedOutput computes outView = scores @ valueView.
*/
func computeWeightedOutput(
	scores, valueView, outView []float32,
	seqQ, seqK, valueDim int,
) {
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
}
