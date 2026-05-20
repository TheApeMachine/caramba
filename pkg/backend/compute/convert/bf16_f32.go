package convert

import "github.com/theapemachine/manifesto/dtype"

/*
BFloat16ToFloat32 converts a slice of BF16 values to float32, writing
into the caller-supplied destination. len(dst) must equal len(src).
Dispatches to the per-architecture SIMD path (bf16_f32_amd64.go or
bf16_f32_arm64.go); falls back to the scalar reference on other GOOS.
*/
func BFloat16ToFloat32(dst []float32, src []dtype.BF16) error {
	return bfloat16ToFloat32(dst, src)
}

/*
Float32ToBFloat16 converts a slice of float32 to BF16, writing into
dst. Truncation rounding matches the hardware BF16 cast intrinsic on
every supported target.
*/
func Float32ToBFloat16(dst []dtype.BF16, src []float32) error {
	return float32ToBFloat16(dst, src)
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
