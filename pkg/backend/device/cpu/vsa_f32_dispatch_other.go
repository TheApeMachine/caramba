//go:build !arm64

package cpu

func VsaBindFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] * right[index]
	}
}

func VsaBundleFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] + right[index]
	}
}

func VsaSimilarityFloat32Native(left, right []float32) float32 {
	var sum float32

	for index := range left {
		sum += left[index] * right[index]
	}

	return sum
}

func VsaPermuteFloat32Native(dst, src []float32, shift int) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	rotation := ((shift % elementCount) + elementCount) % elementCount

	for index, value := range src {
		target := (index + rotation) % elementCount
		dst[target] = value
	}
}
