package attention

import "unsafe"

func ScaledDotProductAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	queryView := unsafe.Slice((*float32)(query), seqQ*depth)
	keyView := unsafe.Slice((*float32)(key), seqK*depth)
	valueView := unsafe.Slice((*float32)(value), seqK*valueDim)
	outputView := unsafe.Slice((*float32)(output), seqQ*valueDim)
	scale := float32(1.0 / float64(depth))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		RunFlashAttentionRowNative(
			queryView, keyView, valueView, outputView,
			rowIndex, seqK, depth, valueDim, scale, config.Causal,
		)
	}
}

func FlashAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
) {
	ScaledDotProductAttention(config, query, key, value, output, seqQ, seqK, depth, valueDim)
}

func MultiHeadAttention(
	config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
) {
	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	queryFeatures := config.NumHeads * config.HeadDim
	kvFeatures := kvHeads * config.HeadDim

	queryView := unsafe.Slice((*float32)(query), seqQ*queryFeatures)
	keyView := unsafe.Slice((*float32)(key), seqK*kvFeatures)
	valueView := unsafe.Slice((*float32)(value), seqK*kvFeatures)
	outputView := unsafe.Slice((*float32)(output), seqQ*queryFeatures)

	multiHeadAttentionSlices(config, queryView, keyView, valueView, outputView, seqQ, seqK, kvHeads)
}
