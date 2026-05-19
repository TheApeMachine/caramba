package neon

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
FlashAttention — block-tiled attention with online softmax. The host
reference here implements the algorithm faithfully but in scalar Go
without block-tiling optimization; the value of the reference is
that it matches the device implementations bit-for-bit so parity
tests in later sessions can validate the device kernels against it.

Args order: (query, key, value, output) with optional mask handling
via a separate FlashAttentionConfig parameter.

Tensor shapes (no batch dim — the batched variant lands in a follow-
up session):
  - query  [seqQ, depth]
  - key    [seqK, depth]
  - value  [seqK, valueDim]
  - output [seqQ, valueDim]

Per Phase 8.2, this is the same forward result as the basic attention
kernel; the difference is the block-tiled execution path that
reduces memory bandwidth on real hardware. The reference here is a
single-block execution to keep the math obvious.
*/

type FlashAttentionConfig struct {
	BlockSize int
	Causal    bool
}

func DefaultFlashAttentionConfig() FlashAttentionConfig {
	return FlashAttentionConfig{BlockSize: 64, Causal: false}
}

func init() {
	Default.Register(Kernel{
		Name: "flash_attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runFlashAttentionFloat32Default,
	})

	Default.Register(Kernel{
		Name: "flash_attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runFlashAttentionBFloat16,
	})

	Default.Register(Kernel{
		Name: "flash_attention",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runFlashAttentionFloat16,
	})
}

func runFlashAttentionBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]
	seqQ, seqK, depth, valueDim, err := attentionDims(query, key, value, out)

	if err != nil {
		return err
	}

	qBF, _ := query.BFloat16Native()
	kBF, _ := key.BFloat16Native()
	vBF, _ := value.BFloat16Native()
	oBF, _ := out.BFloat16Native()

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

	config := DefaultFlashAttentionConfig()
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		RunFlashAttentionRowNative(qF32, kF32, vF32, oF32, rowIndex, seqK, depth, valueDim, scale, config.Causal)
	}

	Float32BulkToBFloat16(oBF, oF32)
	return nil
}

func runFlashAttentionFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	query, key, value, out := args[0], args[1], args[2], args[3]
	seqQ, seqK, depth, valueDim, err := attentionDims(query, key, value, out)

	if err != nil {
		return err
	}

	qF16, _ := query.Float16Native()
	kF16, _ := key.Float16Native()
	vF16, _ := value.Float16Native()
	oF16, _ := out.Float16Native()

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

	config := DefaultFlashAttentionConfig()
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		RunFlashAttentionRowNative(qF32, kF32, vF32, oF32, rowIndex, seqK, depth, valueDim, scale, config.Causal)
	}

	Float32BulkToFloat16(oF16, oF32)
	return nil
}

func runFlashAttentionFloat32Default(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return RunFlashAttentionFloat32(
		DefaultFlashAttentionConfig(),
		args[0], args[1], args[2], args[3],
	)
}

/*
RunFlashAttentionFloat32 runs flash-attention with the supplied
configuration. Causal masking zeros the attention score for any
query→key pair where keyIndex > queryIndex (lower-triangular).
*/
func RunFlashAttentionFloat32(
	config FlashAttentionConfig,
	query, key, value, out tensor.Tensor,
) error {
	queryView, keyView, valueView, outView, seqQ, seqK, depth, valueDim, err :=
		attentionViews(query, key, value, out)

	if err != nil {
		return err
	}

	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		RunFlashAttentionRowNative(
			queryView, keyView, valueView, outView,
			rowIndex, seqK, depth, valueDim, scale, config.Causal,
		)
	}

	return nil
}

func runFlashAttentionRow(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causal bool,
) {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := make([]float32, valueDim)
	scaleScratch := BorrowFloat32Buffer(valueDim)
	valueScratch := BorrowFloat32Buffer(valueDim)

	defer ReleaseFloat32Buffer(scaleScratch)
	defer ReleaseFloat32Buffer(valueScratch)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causal && keyIndex > rowIndex {
			continue
		}

		queryRow := queryView[rowIndex*depth : (rowIndex+1)*depth]
		keyRow := keyView[keyIndex*depth : (keyIndex+1)*depth]
		score := DotFloat32Native(queryRow, keyRow) * scale
		oldMax := maxScore

		if score > maxScore {
			maxScore = score
		}

		alpha := flashExpFloat32(oldMax - maxScore)
		shifted := flashExpFloat32(score - maxScore)
		normalizer = normalizer*alpha + shifted

		fillScaleScratch(scaleScratch, alpha, valueDim)
		MulFloat32Native(accumulator, accumulator, scaleScratch)

		valueRow := valueView[keyIndex*valueDim : (keyIndex+1)*valueDim]
		fillScaleScratch(valueScratch, shifted, valueDim)
		MulFloat32Native(valueScratch, valueScratch, valueRow)
		AddFloat32Native(accumulator, accumulator, valueScratch)
	}

	if normalizer == 0 {
		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			outView[rowIndex*valueDim+dimIndex] = 0
		}

		return
	}

	invNormalizer := float32(1) / normalizer
	fillScaleScratch(scaleScratch, invNormalizer, valueDim)
	MulFloat32Native(
		outView[rowIndex*valueDim:(rowIndex+1)*valueDim],
		accumulator,
		scaleScratch,
	)
}

func flashExpFloat32(value float32) float32 {
	if math.IsInf(float64(value), -1) || value < -88 {
		return 0
	}

	if math.IsInf(float64(value), 1) {
		return float32(math.Inf(1))
	}

	scratch := [1]float32{value}
	ExpFloat32Native(scratch[:], scratch[:])

	return scratch[0]
}

func fillScaleScratch(scratch []float32, value float32, count int) {
	for index := 0; index < count; index++ {
		scratch[index] = value
	}
}

func computeDot(queryView, keyView []float32, rowIndex, keyIndex, depth int) float32 {
	queryRow := queryView[rowIndex*depth : (rowIndex+1)*depth]
	keyRow := keyView[keyIndex*depth : (keyIndex+1)*depth]

	return DotFloat32Native(queryRow, keyRow)
}
