//go:build !amd64 && !arm64

package dequant

import "github.com/theapemachine/manifesto/tensor"

func DequantInt4Native(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	dequantInt4Generic(dst, pairs, scale, zeroPoint)
}
