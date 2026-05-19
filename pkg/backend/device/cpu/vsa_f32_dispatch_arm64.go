//go:build arm64

package cpu

func VsaBindFloat32Native(dst, left, right []float32) {
	MulFloat32Native(dst, left, right)
}

func VsaBundleFloat32Native(dst, left, right []float32) {
	AddFloat32Native(dst, left, right)
}

func VsaSimilarityFloat32Native(left, right []float32) float32 {
	return DotFloat32Native(left, right)
}

func VsaPermuteFloat32Native(dst, src []float32, shift int) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	rotation := ((shift % elementCount) + elementCount) % elementCount
	tailCount := elementCount - rotation

	if tailCount > 0 {
		blockCount := tailCount &^ 3

		if blockCount > 0 {
			VsaPermuteCopyF32NEONAsm(&dst[rotation], &src[0], blockCount)
		}

		for index := blockCount; index < tailCount; index++ {
			dst[rotation+index] = src[index]
		}
	}

	if rotation > 0 {
		blockCount := rotation &^ 3

		if blockCount > 0 {
			VsaPermuteCopyF32NEONAsm(&dst[0], &src[tailCount], blockCount)
		}

		for index := blockCount; index < rotation; index++ {
			dst[index] = src[tailCount+index]
		}
	}
}
