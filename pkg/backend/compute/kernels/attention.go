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

	Default.Register(Kernel{
		Name: "attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAttentionBFloat16,
	})

	Default.Register(Kernel{
		Name: "attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAttentionFloat16,
	})
}

func runAttentionBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]
	seqQ, seqK, depth, valueDim, err := attentionDims(query, key, value, out)

	if err != nil {
		return err
	}

	qBF, err := query.BFloat16Native()

	if err != nil {
		return err
	}

	kBF, err := key.BFloat16Native()

	if err != nil {
		return err
	}

	vBF, err := value.BFloat16Native()

	if err != nil {
		return err
	}

	oBF, err := out.BFloat16Native()

	if err != nil {
		return err
	}

	qF32 := borrowFloat32Buffer(len(qBF))
	kF32 := borrowFloat32Buffer(len(kBF))
	vF32 := borrowFloat32Buffer(len(vBF))
	oF32 := borrowFloat32Buffer(len(oBF))

	defer releaseFloat32Buffer(qF32)
	defer releaseFloat32Buffer(kF32)
	defer releaseFloat32Buffer(vF32)
	defer releaseFloat32Buffer(oF32)

	bfloat16BulkToFloat32(qF32, qBF)
	bfloat16BulkToFloat32(kF32, kBF)
	bfloat16BulkToFloat32(vF32, vBF)

	scale := float32(1.0 / math.Sqrt(float64(depth)))
	scores := computeAttentionScores(qF32, kF32, seqQ, seqK, depth, scale)
	applySoftmax(scores, seqQ, seqK)
	computeWeightedOutput(scores, vF32, oF32, seqQ, seqK, valueDim)

	float32BulkToBFloat16(oBF, oF32)
	return nil
}

func runAttentionFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]
	seqQ, seqK, depth, valueDim, err := attentionDims(query, key, value, out)

	if err != nil {
		return err
	}

	qF16, err := query.Float16Native()

	if err != nil {
		return err
	}

	kF16, err := key.Float16Native()

	if err != nil {
		return err
	}

	vF16, err := value.Float16Native()

	if err != nil {
		return err
	}

	oF16, err := out.Float16Native()

	if err != nil {
		return err
	}

	qF32 := borrowFloat32Buffer(len(qF16))
	kF32 := borrowFloat32Buffer(len(kF16))
	vF32 := borrowFloat32Buffer(len(vF16))
	oF32 := borrowFloat32Buffer(len(oF16))

	defer releaseFloat32Buffer(qF32)
	defer releaseFloat32Buffer(kF32)
	defer releaseFloat32Buffer(vF32)
	defer releaseFloat32Buffer(oF32)

	float16BulkToFloat32(qF32, qF16)
	float16BulkToFloat32(kF32, kF16)
	float16BulkToFloat32(vF32, vF16)

	scale := float32(1.0 / math.Sqrt(float64(depth)))
	scores := computeAttentionScores(qF32, kF32, seqQ, seqK, depth, scale)
	applySoftmax(scores, seqQ, seqK)
	computeWeightedOutput(scores, vF32, oF32, seqQ, seqK, valueDim)

	float32BulkToFloat16(oF16, oF32)
	return nil
}

func attentionDims(query, key, value, out tensor.Tensor) (seqQ, seqK, depth, valueDim int, err error) {
	queryDims := query.Shape().Dims()
	keyDims := key.Shape().Dims()
	valueDims := value.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 ||
		len(valueDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	seqQ = queryDims[0]
	depth = queryDims[1]
	seqK = keyDims[0]
	valueDim = valueDims[1]

	if keyDims[1] != depth || valueDims[0] != seqK ||
		outDims[0] != seqQ || outDims[1] != valueDim {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	return seqQ, seqK, depth, valueDim, nil
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
		queryRow := queryView[rowIndex*depth : (rowIndex+1)*depth]

		for keyIndex := 0; keyIndex < seqK; keyIndex++ {
			keyRow := keyView[keyIndex*depth : (keyIndex+1)*depth]
			scores[rowIndex*seqK+keyIndex] = dotFloat32Native(queryRow, keyRow) * scale
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
		maximum := findRowMax(row)
		sum := fillShiftedExps(row, row, maximum)
		normalizeRow(row, sum)
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
		scoresRow := scores[rowIndex*seqK : (rowIndex+1)*seqK]
		outRow := outView[rowIndex*valueDim : (rowIndex+1)*valueDim]

		for index := range outRow {
			outRow[index] = 0
		}

		matmulFloat32Native(outRow, scoresRow, valueView, 1, seqK, valueDim)
	}
}
