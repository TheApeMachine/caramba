package normalization

import "math/rand"

func randomNormalizationRow(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	row := make([]float32, length)

	for index := range row {
		row[index] = float32((rng.Float64() - 0.5) * 4)
	}

	return row
}
