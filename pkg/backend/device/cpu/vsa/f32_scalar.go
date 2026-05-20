package vsa

/*
VsaBindFloat32Scalar computes out[i] = left[i] * right[i].
*/
func VsaBindFloat32Scalar(out, left, right []float32) {
	for index := range out {
		out[index] = left[index] * right[index]
	}
}

/*
VsaBundleFloat32Scalar computes out[i] = left[i] + right[i].
*/
func VsaBundleFloat32Scalar(out, left, right []float32) {
	for index := range out {
		out[index] = left[index] + right[index]
	}
}

/*
VsaPermuteFloat32Scalar cyclically shifts in by shift into out.
*/
func VsaPermuteFloat32Scalar(out, in []float32, shift int) {
	elementCount := len(in)

	if elementCount == 0 {
		return
	}

	if shift == 0 {
		copy(out, in)
		return
	}

	copy(out[elementCount-shift:], in[:shift])
	copy(out[:elementCount-shift], in[shift:])
}

/*
VsaSimilarityFloat32Scalar is the dot product with f64 accumulation.
*/
func VsaSimilarityFloat32Scalar(left, right []float32) float32 {
	var sum float64

	for index := range left {
		sum += float64(left[index]) * float64(right[index])
	}

	return float32(sum)
}
