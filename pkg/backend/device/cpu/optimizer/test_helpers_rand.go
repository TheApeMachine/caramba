package optimizer

import (
	"math"
	"math/rand"
)

func randFloat32Slice(count int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, count)

	for index := range out {
		out[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return out
}
