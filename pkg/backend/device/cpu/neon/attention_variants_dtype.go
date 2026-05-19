package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Mixed-precision dispatch for multi_head_attention, grouped_query_attention,
and sliding_window_attention. Each follows the standard
widen-bf16/fp16→f32 → run f32 kernel → narrow-f32→bf16/fp16 pattern,
with the operation's actual math (Q@K^T, softmax, @V) running in f32
per §5.5 mixed-dtype convention.
*/

func init() {
	for _, name := range []string{
		"multi_head_attention",
		"grouped_query_attention",
		"sliding_window_attention",
	} {
		name := name
		Default.Register(Kernel{
			Name: name,
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16},
				Outputs: []dtype.DType{dtype.BFloat16},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runMultiHeadAttentionBFloat16(args, configForVariant(name))
			},
		})

		Default.Register(Kernel{
			Name: name,
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.Float16, dtype.Float16, dtype.Float16},
				Outputs: []dtype.DType{dtype.Float16},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runMultiHeadAttentionFloat16(args, configForVariant(name))
			},
		})
	}
}

func configForVariant(name string) MultiHeadAttentionConfig {
	config := DefaultMultiHeadAttentionConfig()

	switch name {
	case "grouped_query_attention":
		config.KVHeadCount = config.NumHeads / 4
	case "sliding_window_attention":
		config.Causal = true
		config.WindowSize = 128
	}

	return config
}

func multiHeadAttentionMixedDims(
	config MultiHeadAttentionConfig,
	query, key, value, out tensor.Tensor,
) (seqQ, seqK, kvHeads int, err error) {
	queryDims := query.Shape().Dims()
	keyDims := key.Shape().Dims()
	valueDims := value.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 ||
		len(valueDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	kvHeads = config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	queryFeatures := config.NumHeads * config.HeadDim
	kvFeatures := kvHeads * config.HeadDim

	if queryDims[1] != queryFeatures || keyDims[1] != kvFeatures ||
		valueDims[1] != kvFeatures || outDims[1] != queryFeatures {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	seqQ = queryDims[0]
	seqK = keyDims[0]

	if valueDims[0] != seqK || outDims[0] != seqQ {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return seqQ, seqK, kvHeads, nil
}

func runMultiHeadAttentionBFloat16(args []tensor.Tensor, config MultiHeadAttentionConfig) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	seqQ, seqK, kvHeads, err := multiHeadAttentionMixedDims(config, args[0], args[1], args[2], args[3])

	if err != nil {
		return err
	}

	qBF, err := args[0].BFloat16Native()
	if err != nil {
		return err
	}

	kBF, err := args[1].BFloat16Native()
	if err != nil {
		return err
	}

	vBF, err := args[2].BFloat16Native()
	if err != nil {
		return err
	}

	oBF, err := args[3].BFloat16Native()
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

	multiHeadAttentionSlices(config, qF32, kF32, vF32, oF32, seqQ, seqK, kvHeads)

	Float32BulkToBFloat16(oBF, oF32)
	return nil
}

func runMultiHeadAttentionFloat16(args []tensor.Tensor, config MultiHeadAttentionConfig) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	seqQ, seqK, kvHeads, err := multiHeadAttentionMixedDims(config, args[0], args[1], args[2], args[3])

	if err != nil {
		return err
	}

	qF16, err := args[0].Float16Native()
	if err != nil {
		return err
	}

	kF16, err := args[1].Float16Native()
	if err != nil {
		return err
	}

	vF16, err := args[2].Float16Native()
	if err != nil {
		return err
	}

	oF16, err := args[3].Float16Native()
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

	multiHeadAttentionSlices(config, qF32, kF32, vF32, oF32, seqQ, seqK, kvHeads)

	Float32BulkToFloat16(oF16, oF32)
	return nil
}
