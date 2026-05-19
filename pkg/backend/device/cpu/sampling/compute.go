package sampling

import (
	"math/rand/v2"
	"sort"
)

func softmaxAndSort(logits []float32, temperature float32) ([]float32, []int) {
	probabilities := make([]float32, len(logits))
	indices := make([]int, len(logits))

	SamplingSoftmaxRowFloat32Native(logits, probabilities, temperature)

	for index := range indices {
		indices[index] = index
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
