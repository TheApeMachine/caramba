package sampling

import (
	"math"
	"math/rand"
)

func randomSamplingLogits(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	logits := make([]float32, length)

	for index := range logits {
		logits[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return logits
}
