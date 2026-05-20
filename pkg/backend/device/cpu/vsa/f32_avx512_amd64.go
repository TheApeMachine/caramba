//go:build amd64

package vsa

//go:noescape
func VsaBindFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func VsaBundleFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func VsaPermuteCopyFloat32AVX512Asm(dst, src *float32, count int)

//go:noescape
func VsaSimilarityFloat32AVX512Asm(left, right *float32, count int) float32

func VsaBindF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	VsaBindFloat32AVX512Asm(dst, left, right, count)
}

func VsaBundleF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	VsaBundleFloat32AVX512Asm(dst, left, right, count)
}

func VsaPermuteCopyF32AVX512(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	VsaPermuteCopyFloat32AVX512Asm(dst, src, count)
}

func VsaSimilarityF32AVX512(left, right *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return VsaSimilarityFloat32AVX512Asm(left, right, count)
}
