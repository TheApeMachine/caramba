//go:build arm64

package kernels

//go:noescape
func vsaPermuteCopyF32NEONAsm(dst, src *float32, n int)

func vsaBindFloat32Native(dst, left, right []float32) {
	mulFloat32Native(dst, left, right)
}

func vsaBundleFloat32Native(dst, left, right []float32) {
	addFloat32Native(dst, left, right)
}

func vsaSimilarityFloat32Native(left, right []float32) float32 {
	return dotFloat32Native(left, right)
}

func vsaPermuteFloat32Native(dst, src []float32, shift int) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	rotation := ((shift % elementCount) + elementCount) % elementCount
	tailCount := elementCount - rotation

	if tailCount > 0 {
		blockCount := tailCount &^ 3

		if blockCount > 0 {
			vsaPermuteCopyF32NEONAsm(&dst[rotation], &src[0], blockCount)
		}

		for index := blockCount; index < tailCount; index++ {
			dst[rotation+index] = src[index]
		}
	}

	if rotation > 0 {
		blockCount := rotation &^ 3

		if blockCount > 0 {
			vsaPermuteCopyF32NEONAsm(&dst[0], &src[tailCount], blockCount)
		}

		for index := blockCount; index < rotation; index++ {
			dst[index] = src[tailCount+index]
		}
	}
}
