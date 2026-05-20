//go:build amd64

package vsa

import "golang.org/x/sys/cpu"

func VsaBindFloat32Native(out, left, right []float32) {
	if len(out) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		VsaBindF32AVX512(&out[0], &left[0], &right[0], len(out))
		return
	}

	VsaBindFloat32Scalar(out, left, right)
}

func VsaBundleFloat32Native(out, left, right []float32) {
	if len(out) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		VsaBundleF32AVX512(&out[0], &left[0], &right[0], len(out))
		return
	}

	VsaBundleFloat32Scalar(out, left, right)
}

func VsaPermuteFloat32Native(out, in []float32, shift int) {
	elementCount := len(in)

	if elementCount == 0 {
		return
	}

	if shift == 0 {
		copyBlockNative(out, in)
		return
	}

	firstLength := elementCount - shift
	secondLength := shift

	if firstLength > 0 {
		copyBlockNative(out[:firstLength], in[shift:shift+firstLength])
	}

	if secondLength > 0 {
		copyBlockNative(out[firstLength:], in[:secondLength])
	}
}

func VsaSimilarityFloat32Native(left, right []float32) float32 {
	if len(left) == 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		return VsaSimilarityF32AVX512(&left[0], &right[0], len(left))
	}

	return VsaSimilarityFloat32Scalar(left, right)
}

func copyBlockNative(dst, src []float32) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		VsaPermuteCopyF32AVX512(&dst[0], &src[0], elementCount)
		return
	}

	for index := range src {
		dst[index] = src[index]
	}
}
