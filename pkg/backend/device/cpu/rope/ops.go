package rope

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func RoPE(
	config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	dispatchRoPE(config, input, output, seqLen, numHeads, headDim, format)
}

func RoPEPairs(
	output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
) {
	RopePairsNative(
		unsafe.Slice((*float32)(output), halfDim*2),
		unsafe.Slice((*float32)(input), halfDim*2),
		unsafe.Slice((*float32)(cosBuffer), halfDim),
		unsafe.Slice((*float32)(sinBuffer), halfDim),
	)
}
