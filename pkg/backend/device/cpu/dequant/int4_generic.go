package dequant

import "github.com/theapemachine/manifesto/tensor"

func dequantInt4Generic(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	for index := range dst {
		nibble := pairs.Get(index)
		dst[index] = float32(int(nibble)-int(zeroPoint)) * scale
	}
}
