package masking

import (
	"math"
	"math/rand"
)

func randomMaskingFloat32(length int, seed int64) []float32 {
	randomSource := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = randomSource.Float32()*4 - 2
	}

	return values
}

func randomMaskingScores(seqQ, seqK int, seed int64) []float32 {
	randomSource := rand.New(rand.NewSource(seed))
	values := make([]float32, seqQ*seqK)

	for index := range values {
		values[index] = randomSource.Float32()*2 - 1
	}

	return values
}

func maskingSquareSide(length int) int {
	side := int(math.Sqrt(float64(length)))

	if side < 1 {
		return 1
	}

	return side
}
