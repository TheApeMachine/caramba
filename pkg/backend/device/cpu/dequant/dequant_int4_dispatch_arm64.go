//go:build arm64

package dequant

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func DequantInt4Native(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	if len(dst) == 0 {
		return
	}

	bytes := pairs.Bytes()

	DequantInt4NEONAsm(&dst[0], &bytes[0], len(dst), scale, zeroPoint)
}
