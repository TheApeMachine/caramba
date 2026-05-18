package kernels

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

	config := DefaultFlashAttentionConfig()
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		runFlashAttentionRow(qF32, kF32, vF32, oF32, rowIndex, seqK, depth, valueDim, scale, config.Causal)
	}

	float32BulkToBFloat16(oBF, oF32)
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

	config := DefaultFlashAttentionConfig()
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		runFlashAttentionRow(qF32, kF32, vF32, oF32, rowIndex, seqK, depth, valueDim, scale, config.Causal)
	}

	float32BulkToFloat16(oF16, oF32)
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
		runFlashAttentionRow(
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
	// Online softmax with running max and running normalizer.
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := make([]float32, valueDim)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causal && keyIndex > rowIndex {
			continue
		}

		dot := computeDot(queryView, keyView, rowIndex, keyIndex, depth)
		score := dot * scale

		oldMax := maxScore

		if score > maxScore {
			maxScore = score
		}

		alpha := float32(math.Exp(float64(oldMax - maxScore)))
		shifted := float32(math.Exp(float64(score - maxScore)))
		normalizer = normalizer*alpha + shifted

		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			accumulator[dimIndex] = accumulator[dimIndex]*alpha +
				shifted*valueView[keyIndex*valueDim+dimIndex]
		}
	}

	if normalizer == 0 {
		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			outView[rowIndex*valueDim+dimIndex] = 0
		}

		return
	}

	for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
		outView[rowIndex*valueDim+dimIndex] = accumulator[dimIndex] / normalizer
	}
}

func computeDot(queryView, keyView []float32, rowIndex, keyIndex, depth int) float32 {
	var dot float32

	for depthIndex := 0; depthIndex < depth; depthIndex++ {
		dot += queryView[rowIndex*depth+depthIndex] *
			keyView[keyIndex*depth+depthIndex]
	}

	return dot
}
