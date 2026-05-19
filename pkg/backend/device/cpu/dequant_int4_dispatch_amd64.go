//go:build amd64

package cpu

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func DequantInt4Native(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	for index := range dst {
		nibble := pairs.Get(index)
		dst[index] = float32(int(nibble)-int(zeroPoint)) * scale
	}
}
