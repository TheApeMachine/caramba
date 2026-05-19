package sampling

import (
	"math"
	"math/rand/v2"
	"sort"
)

func softmaxAndSort(logits []float32, temperature float32) ([]float32, []int) {
	if temperature == 0 {
		temperature = 1
	}

	probabilities := make([]float32, len(logits))
	indices := make([]int, len(logits))

	maximum := logits[0]

	for _, value := range logits[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var denominator float64

	for index, value := range logits {
		shifted := math.Exp(float64((value - maximum) / temperature))
		probabilities[index] = float32(shifted)
		indices[index] = index
		denominator += shifted
	}

	scale := float32(1.0 / denominator)

	for index := range probabilities {
		probabilities[index] *= scale
	}

	sort.SliceStable(indices, func(left, right int) bool {
		return probabilities[indices[left]] > probabilities[indices[right]]
	})

	sorted := make([]float32, len(probabilities))

	for resultIndex, originalIndex := range indices {
		sorted[resultIndex] = probabilities[originalIndex]
	}

	return sorted, indices
}

func newSamplingRNG(seed uint64) *rand.Rand {
	source := rand.NewChaCha8([32]byte{
		byte(seed), byte(seed >> 8), byte(seed >> 16), byte(seed >> 24),
		byte(seed >> 32), byte(seed >> 40), byte(seed >> 48), byte(seed >> 56),
	})

	return rand.New(source)
}
