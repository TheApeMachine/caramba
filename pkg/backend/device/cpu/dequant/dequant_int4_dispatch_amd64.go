//go:build amd64

package dequant

import (
	"github.com/theapemachine/manifesto/tensor"
	"golang.org/x/sys/cpu"
)

func DequantInt4Native(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	elementCount := len(dst)

	if elementCount == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		dequantInt4AVX512(dst, pairs.Bytes(), elementCount, scale, zeroPoint)

		return
	}

	dequantInt4Generic(dst, pairs, scale, zeroPoint)
}
