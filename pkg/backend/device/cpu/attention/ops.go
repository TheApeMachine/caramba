package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func ScaledDotProductAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	dispatchScaledDotProductAttention(
		config, query, key, value, output,
		seqQ, seqK, depth, valueDim, format,
	)
}

func FlashAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	ScaledDotProductAttention(
		config, query, key, value, output,
		seqQ, seqK, depth, valueDim, format,
	)
}

func MultiHeadAttention(
	config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	dispatchMultiHeadAttention(config, query, key, value, output, seqQ, seqK, format)
}
