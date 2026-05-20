package attention

import (
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

/*
ScaledDotProductAttention is the host reference for the canonical
attention kernel: softmax(Q @ K^T / sqrt(d_k)) @ V.
*/

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

	qF32 := BorrowFloat32Buffer(len(qBF))
	kF32 := BorrowFloat32Buffer(len(kBF))
	vF32 := BorrowFloat32Buffer(len(vBF))
	oF32 := BorrowFloat32Buffer(len(oBF))

	defer ReleaseFloat32Buffer(qF32)
	defer ReleaseFloat32Buffer(kF32)
	defer ReleaseFloat32Buffer(vF32)
	defer ReleaseFloat32Buffer(oF32)

	Bfloat16BulkToFloat32(qF32, qBF)
	Bfloat16BulkToFloat32(kF32, kBF)
	Bfloat16BulkToFloat32(vF32, vBF)

	scale := float32(1.0 / math.Sqrt(float64(depth)))
	scores := computeAttentionScores(qF32, kF32, seqQ, seqK, depth, scale)
	applySoftmax(scores, seqQ, seqK)
	computeWeightedOutput(scores, vF32, oF32, seqQ, seqK, valueDim)

	Float32BulkToBFloat16(oBF, oF32)
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

	qF32 := BorrowFloat32Buffer(len(qF16))
	kF32 := BorrowFloat32Buffer(len(kF16))
	vF32 := BorrowFloat32Buffer(len(vF16))
	oF32 := BorrowFloat32Buffer(len(oF16))

	defer ReleaseFloat32Buffer(qF32)
	defer ReleaseFloat32Buffer(kF32)
	defer ReleaseFloat32Buffer(vF32)
	defer ReleaseFloat32Buffer(oF32)

	Float16BulkToFloat32(qF32, qF16)
	Float16BulkToFloat32(kF32, kF16)
	Float16BulkToFloat32(vF32, vF16)

	scale := float32(1.0 / math.Sqrt(float64(depth)))
	scores := computeAttentionScores(qF32, kF32, seqQ, seqK, depth, scale)
	applySoftmax(scores, seqQ, seqK)
	computeWeightedOutput(scores, vF32, oF32, seqQ, seqK, valueDim)

	Float32BulkToFloat16(oF16, oF32)
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

	return RunFlashAttentionFloat32(
		DefaultFlashAttentionConfig(),
		query, key, value, out,
	)
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
			scores[rowIndex*seqK+keyIndex] = DotFloat32Native(queryRow, keyRow) * scale
		}
	}

	return scores
}

/*
applySoftmax performs row-wise stable softmax in place on a
[seqQ, seqK] score matrix.
*/
func applySoftmax(scores []float32, seqQ, seqK int) {
	ApplyAttentionSoftmaxNative(scores, seqQ, seqK)
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

		MatmulFloat32Native(outRow, scoresRow, valueView, 1, seqK, valueDim)
	}
}
