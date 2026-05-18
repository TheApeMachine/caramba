package convert

import "github.com/theapemachine/caramba/pkg/dtype"

/*
Float16ToFloat32 converts a slice of IEEE 754 binary16 values to
float32.
*/
func Float16ToFloat32(dst []float32, src []dtype.F16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = value.Float32()
	}

	return nil
}

/*
Float32ToFloat16 converts a slice of float32 to F16 using IEEE 754
round-to-nearest-even.
*/
func Float32ToFloat16(dst []dtype.F16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = dtype.Fromfloat32(value)
	}

	return nil
}
