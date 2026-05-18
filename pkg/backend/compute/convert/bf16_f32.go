package convert

import "github.com/theapemachine/caramba/pkg/dtype"

/*
BFloat16ToFloat32 converts a slice of BF16 values to float32, writing
into the caller-supplied destination. len(dst) must equal len(src).
*/
func BFloat16ToFloat32(dst []float32, src []dtype.BF16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	return bfloat16ToFloat32Scalar(dst, src)
}

/*
Float32ToBFloat16 converts a slice of float32 to BF16, writing into
dst. Truncation rounding matches the hardware BF16 cast intrinsic on
every supported target.
*/
func Float32ToBFloat16(dst []dtype.BF16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	return float32ToBFloat16Scalar(dst, src)
}

func bfloat16ToFloat32Scalar(dst []float32, src []dtype.BF16) error {
	for index, value := range src {
		dst[index] = (&value).Float32()
	}

	return nil
}

func float32ToBFloat16Scalar(dst []dtype.BF16, src []float32) error {
	for index, value := range src {
		dst[index] = dtype.NewBfloat16FromFloat32(value)
	}

	return nil
}
