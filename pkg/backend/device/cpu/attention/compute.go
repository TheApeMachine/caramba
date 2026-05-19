package attention

import (
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func dispatchScaledDotProductAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		scaledDotProductAttentionF32(
			config, query, key, value, output,
			seqQ, seqK, depth, valueDim,
		)
	case dtype.Float16, dtype.BFloat16:
		scaledDotProductAttentionMixed(
			config, query, key, value, output,
			seqQ, seqK, depth, valueDim, format,
		)
	default:
		panic("attention: ScaledDotProductAttention unsupported dtype")
	}
}

func dispatchMultiHeadAttention(
	config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	queryFeatures := config.NumHeads * config.HeadDim
	kvFeatures := kvHeads * config.HeadDim

	switch format {
	case dtype.Float32:
		queryView := unsafe.Slice((*float32)(query), seqQ*queryFeatures)
		keyView := unsafe.Slice((*float32)(key), seqK*kvFeatures)
		valueView := unsafe.Slice((*float32)(value), seqK*kvFeatures)
		outputView := unsafe.Slice((*float32)(output), seqQ*queryFeatures)
		multiHeadAttentionSlices(config, queryView, keyView, valueView, outputView, seqQ, seqK, kvHeads)
	case dtype.Float16, dtype.BFloat16:
		multiHeadAttentionMixed(config, query, key, value, output, seqQ, seqK, kvHeads, format)
	default:
		panic("attention: MultiHeadAttention unsupported dtype")
	}
}

func scaledDotProductAttentionMixed(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	queryCount := seqQ * depth
	keyCount := seqK * depth
	valueCount := seqK * valueDim
	outputCount := seqQ * valueDim

	queryF32 := BorrowFloat32Buffer(queryCount)
	keyF32 := BorrowFloat32Buffer(keyCount)
	valueF32 := BorrowFloat32Buffer(valueCount)
	outputF32 := BorrowFloat32Buffer(outputCount)

	defer ReleaseFloat32Buffer(queryF32)
	defer ReleaseFloat32Buffer(keyF32)
	defer ReleaseFloat32Buffer(valueF32)
	defer ReleaseFloat32Buffer(outputF32)

	widenAttentionBuffer(query, queryF32, format)
	widenAttentionBuffer(key, keyF32, format)
	widenAttentionBuffer(value, valueF32, format)

	scaledDotProductAttentionF32(
		config,
		unsafe.Pointer(&queryF32[0]),
		unsafe.Pointer(&keyF32[0]),
		unsafe.Pointer(&valueF32[0]),
		unsafe.Pointer(&outputF32[0]),
		seqQ, seqK, depth, valueDim,
	)

	narrowAttentionBuffer(output, outputF32, format)
}

func widenAttentionBuffer(source unsafe.Pointer, destination []float32, format dtype.DType) {
	switch format {
	case dtype.Float16:
		sourceView := unsafe.Slice((*dtype.F16)(source), len(destination))
		Float16BulkToFloat32(destination, sourceView)
	case dtype.BFloat16:
		sourceView := unsafe.Slice((*dtype.BF16)(source), len(destination))
		Bfloat16BulkToFloat32(destination, sourceView)
	}
}

func narrowAttentionBuffer(destination unsafe.Pointer, source []float32, format dtype.DType) {
	switch format {
	case dtype.Float16:
		destinationView := unsafe.Slice((*dtype.F16)(destination), len(source))
		Float32BulkToFloat16(destinationView, source)
	case dtype.BFloat16:
		destinationView := unsafe.Slice((*dtype.BF16)(destination), len(source))
		Float32BulkToBFloat16(destinationView, source)
	}
}

func multiHeadAttentionMixed(
	config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, kvHeads int,
	format dtype.DType,
) {
	queryFeatures := config.NumHeads * config.HeadDim
	kvFeatures := kvHeads * config.HeadDim
	queryCount := seqQ * queryFeatures
	keyCount := seqK * kvFeatures
	valueCount := seqK * kvFeatures
	outputCount := seqQ * queryFeatures

	queryF32 := BorrowFloat32Buffer(queryCount)
	keyF32 := BorrowFloat32Buffer(keyCount)
	valueF32 := BorrowFloat32Buffer(valueCount)
	outputF32 := BorrowFloat32Buffer(outputCount)

	defer ReleaseFloat32Buffer(queryF32)
	defer ReleaseFloat32Buffer(keyF32)
	defer ReleaseFloat32Buffer(valueF32)
	defer ReleaseFloat32Buffer(outputF32)

	widenAttentionBuffer(query, queryF32, format)
	widenAttentionBuffer(key, keyF32, format)
	widenAttentionBuffer(value, valueF32, format)

	multiHeadAttentionSlices(config, queryF32, keyF32, valueF32, outputF32, seqQ, seqK, kvHeads)
	narrowAttentionBuffer(output, outputF32, format)
}

func scaledDotProductAttentionF32(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
) {
	queryView := unsafe.Slice((*float32)(query), seqQ*depth)
	keyView := unsafe.Slice((*float32)(key), seqK*depth)
	valueView := unsafe.Slice((*float32)(value), seqK*valueDim)
	outputView := unsafe.Slice((*float32)(output), seqQ*valueDim)
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		RunFlashAttentionRowNative(
			queryView, keyView, valueView, outputView,
			rowIndex, seqK, depth, valueDim, scale, config.Causal,
		)
	}
}
