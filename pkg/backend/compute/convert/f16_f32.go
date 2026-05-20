package convert

import "github.com/theapemachine/manifesto/dtype"

/*
Float16ToFloat32 converts a slice of IEEE 754 binary16 values to
float32. Dispatches to the per-architecture native path (arm64 NEON,
amd64 F16C) and falls back to the scalar reference otherwise.
*/
func Float16ToFloat32(dst []float32, src []dtype.F16) error {
	return float16ToFloat32Native(dst, src)
}

/*
Float32ToFloat16 converts a slice of float32 to F16 using IEEE 754
round-to-nearest-even.
*/
func Float32ToFloat16(dst []dtype.F16, src []float32) error {
	return float32ToFloat16Native(dst, src)
}

/*
float16ToFloat32Scalar is the scalar reference body. The per-arch
files in f16_f32_{arm64,amd64,other}.go override the
*Native dispatchers with SIMD paths; the scalar form here is used by
those overrides when their preconditions are not met.
*/
func float16ToFloat32Scalar(dst []float32, src []dtype.F16) error {
	for index, value := range src {
		dst[index] = value.Float32()
	}

	return nil
}

func float32ToFloat16Scalar(dst []dtype.F16, src []float32) error {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(value)
	}

	return nil
}
