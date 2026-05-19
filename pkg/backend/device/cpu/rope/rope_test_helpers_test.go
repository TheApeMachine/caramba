package rope

import "math/rand"

func randomRopePairBuffers(pairCount int, seed int64) (in, cos, sin []float32) {
	rng := rand.New(rand.NewSource(seed))
	in = make([]float32, 2*pairCount)
	cos = make([]float32, pairCount)
	sin = make([]float32, pairCount)

	for pairIndex := 0; pairIndex < pairCount; pairIndex++ {
		in[2*pairIndex] = float32((rng.Float64() - 0.5) * 4)
		in[2*pairIndex+1] = float32((rng.Float64() - 0.5) * 4)
		cos[pairIndex] = float32((rng.Float64() - 0.5) * 2)
		sin[pairIndex] = float32((rng.Float64() - 0.5) * 2)
	}

	return in, cos, sin
}
