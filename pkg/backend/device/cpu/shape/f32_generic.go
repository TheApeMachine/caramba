package shape

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func copyContiguousF32Generic(dst, src *float32, count int) {
	CopyContiguousGeneric(unsafe.Slice(dst, count), unsafe.Slice(src, count))
}

func whereF32Generic(dst, positive, negative *float32, mask []byte, count int) {
	WhereGeneric(
		unsafe.Slice(dst, count),
		unsafe.Slice(positive, count),
		unsafe.Slice(negative, count),
		mask,
	)
}

func maskedFillF32Generic(dst, input *float32, fill float32, mask []byte, count int) {
	MaskedFillGeneric(
		unsafe.Slice(dst, count),
		unsafe.Slice(input, count),
		fill,
		mask,
	)
}

func CopyContiguousGeneric(dst, src []float32) {
	copy(dst, src)
}

func WhereGeneric(dst, positive, negative []float32, mask []byte) {
	if len(dst) != len(positive) ||
		len(dst) != len(negative) ||
		len(mask)*8 < len(dst) {
		panic("shape: where length mismatch")
	}

	for index := range dst {
		if maskBitAt(mask, index) {
			dst[index] = positive[index]
			continue
		}

		dst[index] = negative[index]
	}
}

func MaskedFillGeneric(dst, input []float32, fill float32, mask []byte) {
	if len(dst) != len(input) || len(mask)*8 < len(input) {
		panic("shape: masked fill length mismatch")
	}

	for index := range dst {
		dst[index] = input[index]

		if maskBitAt(mask, index) {
			dst[index] = fill
		}
	}
}

func maskBitAt(mask []byte, index int) bool {
	return mask[index/8]&(1<<(uint(index)%8)) != 0
}

func bitVectorMaskBytes(mask tensor.BitVector) []byte {
	byteCount := (mask.Len() + 7) / 8

	return mask.Bytes()[:byteCount]
}
