package convolution

/*
ConvPatchDotScalar is the float32 patch dot used by convolution scalar
references: sum(float64(weight[i])*float64(patch[i])) narrowed to float32.
*/
func ConvPatchDotScalar(weight, patch []float32, length int) float32 {
	var sum float64

	for index := range length {
		sum += float64(weight[index]) * float64(patch[index])
	}

	return float32(sum)
}
