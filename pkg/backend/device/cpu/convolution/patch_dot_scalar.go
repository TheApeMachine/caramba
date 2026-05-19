package convolution

/*
ConvPatchDotScalar is the float32 sequential dot product used by
convolution scalar references and parity tests for patch-dot NEON.
*/
func ConvPatchDotScalar(weight, patch []float32, length int) float32 {
	var sum float32

	for index := range length {
		sum += weight[index] * patch[index]
	}

	return sum
}
