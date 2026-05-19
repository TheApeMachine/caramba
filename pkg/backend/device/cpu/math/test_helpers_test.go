package math

import (
	"math"
	"math/rand"
)

func randomMathFloat32(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return values
}

func mathSquareSide(length int) int {
	side := int(math.Sqrt(float64(length)))

	if side < 1 {
		side = 1
	}

	return side
}
