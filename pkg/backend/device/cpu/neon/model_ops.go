package neon

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Model-level operations: LoRA adapter add, weight freezing, model
graft (replace a weight tensor in place), random initialization.

  - lora_apply: y = baseOut + scale × (A × B × x), where A, B are
    the low-rank decomposition matrices and x is the input
    activation.
  - lora_merge:  combine baseWeight + scale × A × B into a single
    weight tensor.
  - weight_freeze_mask: builds a boolean mask for selective weight
    freezing (true → trainable, false → frozen).
*/

type LoRAConfig struct {
	Scale float32
	Rank  int
}

func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{Scale: 1.0, Rank: 8}
}

func runLoRAApplyDefault(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return LoRAApplyFloat32(DefaultLoRAConfig(), args[0], args[1], args[2], args[3], args[4])
}

/*
LoRAApplyFloat32 computes y = baseOut + scale × (loraA × loraB × x).
Args: (baseOut [batch, outDim], loraA [outDim, rank],
loraB [rank, inDim], x [batch, inDim], output [batch, outDim]).
*/
func LoRAApplyFloat32(
	config LoRAConfig,
	baseOut, loraA, loraB, input, output tensor.Tensor,
) error {
	baseView, _ := baseOut.Float32Native()
	aView, _ := loraA.Float32Native()
	bView, _ := loraB.Float32Native()
	inputView, _ := input.Float32Native()
	outView, _ := output.Float32Native()

	aDims := loraA.Shape().Dims()
	bDims := loraB.Shape().Dims()
	xDims := input.Shape().Dims()

	if len(aDims) != 2 || len(bDims) != 2 || len(xDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	outDim := aDims[0]
	rank := aDims[1]
	inDim := bDims[1]
	batch := xDims[0]

	if bDims[0] != rank || xDims[1] != inDim ||
		len(baseView) != batch*outDim || len(outView) != batch*outDim {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		intermediate := make([]float32, rank)

		for rankIndex := 0; rankIndex < rank; rankIndex++ {
			var sum float32

			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += bView[rankIndex*inDim+inIndex] *
					inputView[batchIndex*inDim+inIndex]
			}

			intermediate[rankIndex] = sum
		}

		for outIndex := 0; outIndex < outDim; outIndex++ {
			var sum float32

			for rankIndex := 0; rankIndex < rank; rankIndex++ {
				sum += aView[outIndex*rank+rankIndex] * intermediate[rankIndex]
			}

			outView[batchIndex*outDim+outIndex] =
				baseView[batchIndex*outDim+outIndex] + config.Scale*sum
		}
	}

	return nil
}

func runLoRAMergeDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return LoRAMergeFloat32(DefaultLoRAConfig(), args[0], args[1], args[2], args[3])
}

/*
LoRAMergeFloat32 produces a merged weight tensor:
mergedWeight = baseWeight + scale × loraA × loraB. Useful for
deploying a LoRA-tuned model without keeping A and B separate.
*/
func LoRAMergeFloat32(
	config LoRAConfig,
	baseWeight, loraA, loraB, output tensor.Tensor,
) error {
	baseView, _ := baseWeight.Float32Native()
	aView, _ := loraA.Float32Native()
	bView, _ := loraB.Float32Native()
	outView, _ := output.Float32Native()

	aDims := loraA.Shape().Dims()
	bDims := loraB.Shape().Dims()
	baseDims := baseWeight.Shape().Dims()

	if len(aDims) != 2 || len(bDims) != 2 || len(baseDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	outDim := aDims[0]
	rank := aDims[1]
	inDim := bDims[1]

	if bDims[0] != rank || baseDims[0] != outDim || baseDims[1] != inDim ||
		len(outView) != outDim*inDim {
		return tensor.ErrShapeMismatch
	}

	copy(outView, baseView)

	for outIndex := 0; outIndex < outDim; outIndex++ {
		for inIndex := 0; inIndex < inDim; inIndex++ {
			var update float32

			for rankIndex := 0; rankIndex < rank; rankIndex++ {
				update += aView[outIndex*rank+rankIndex] *
					bView[rankIndex*inDim+inIndex]
			}

			outView[outIndex*inDim+inIndex] += config.Scale * update
		}
	}

	return nil
}

/*
runWeightFreezeMask zeros entries of an input gradient where the mask
is false. Used to freeze parameters during fine-tuning.
*/
func runWeightFreezeMask(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	mask, _ := args[0].BoolNative()
	gradients, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if mask.Len() != len(gradients) || len(out) != len(gradients) {
		return tensor.ErrShapeMismatch
	}

	for index, value := range gradients {
		out[index] = 0

		if mask.Get(index) {
			out[index] = value
		}
	}

	return nil
}
