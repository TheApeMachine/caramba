package convert

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Int8ToFloat32 widens int8 values to float32 directly. For the
parameterized dequantization path (apply a scale factor and zero
point), see pkg/backend/compute/kernels.
*/
func Int8ToFloat32(dst []float32, src []int8) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float32(value)
	}

	return nil
}

/*
Float32ToInt8 narrows float32 to int8 with saturation at
math.MinInt8 / math.MaxInt8.
*/
func Float32ToInt8(dst []int8, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		if value < float32(math.MinInt8) {
			dst[index] = math.MinInt8
			continue
		}

		if value > float32(math.MaxInt8) {
			dst[index] = math.MaxInt8
			continue
		}

		dst[index] = int8(value)
	}

	return nil
}

/*
Int4ToFloat32 widens a packed Int4 byte stream to float32. The source
is a slice of Int4Pair (each holding two sign-extended nibbles);
len(dst) must equal 2 * len(src) (or 2 * len(src) - 1 when the
trailing high nibble is unused — caller passes the correct logical
count via slice length on dst).
*/
func Int4ToFloat32(dst []float32, src []dtype.Int4Pair) error {
	if len(dst) > len(src)*2 {
		return errLenMismatch
	}

	for index := range dst {
		pair := src[index/2]

		if index%2 == 0 {
			dst[index] = float32(pair.Lo())
			continue
		}

		dst[index] = float32(pair.Hi())
	}

	return nil
}
