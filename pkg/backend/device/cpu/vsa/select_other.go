//go:build !arm64

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

	copy(out[elementCount-shift:], in[:shift])
	copy(out[:elementCount-shift], in[shift:])
}

func VsaSimilarityFloat32Native(left, right []float32) float32 {
	return dot.DotFloat32Native(left, right)
}
