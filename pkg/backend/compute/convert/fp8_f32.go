package convert

import "github.com/theapemachine/manifesto/dtype"

/*
Float8E4M3ToFloat32 widens FP8 E4M3 values to float32.
*/
func Float8E4M3ToFloat32(dst []float32, src []dtype.F8E4M3) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = value.Float32()
	}

	return nil
}

/*
Float32ToFloat8E4M3 narrows float32 to FP8 E4M3 using saturating
round-to-nearest-even.
*/
func Float32ToFloat8E4M3(dst []dtype.F8E4M3, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = dtype.NewF8E4M3FromFloat32(value)
	}

	return nil
}

/*
Float8E5M2ToFloat32 widens FP8 E5M2 values to float32.
*/
func Float8E5M2ToFloat32(dst []float32, src []dtype.F8E5M2) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = value.Float32()
	}

	return nil
}

/*
Float32ToFloat8E5M2 narrows float32 to FP8 E5M2 using saturating
round-to-nearest-even.
*/
func Float32ToFloat8E5M2(dst []dtype.F8E5M2, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = dtype.NewF8E5M2FromFloat32(value)
	}

	return nil
}
