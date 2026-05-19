package shape

import (
	"math/rand"
)

func randomShapeFloat32(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = float32((rng.Float64() - 0.5) * 4)
	}

	return values
}

func shapeMaskBytes(length int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	byteCount := (length + 7) / 8
	mask := make([]byte, byteCount)

	for index := range length {
		if rng.Intn(2) == 0 {
			continue
		}

		mask[index/8] |= 1 << (uint(index) % 8)
	}

	return mask
}
