//go:build arm64

package vsa

import (
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dot"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/elementwise"
)

func VsaBindFloat32Native(out, left, right []float32) {
	elementwise.MulFloat32Native(out, left, right)
}

func VsaBundleFloat32Native(out, left, right []float32) {
	elementwise.AddFloat32Native(out, left, right)
}

func VsaPermuteFloat32Native(out, in []float32, shift int) {
	elementCount := len(in)

	if elementCount == 0 {
		return
	}

	if shift == 0 {
		copy(out, in)
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
	return dot.DotFloat32Native(left, right)
}

func copyBlockNative(dst, src []float32) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	blockCount := elementCount &^ 3

	if blockCount > 0 {
		VsaPermuteCopyF32NEONAsm(&dst[0], &src[0], blockCount)
	}

	for index := blockCount; index < elementCount; index++ {
		dst[index] = src[index]
	}
}
