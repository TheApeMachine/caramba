//go:build !arm64

package kernels

import "math"

func greedySampleFloat32Native(logits []float32) int32 {
	maxIndex := 0
	maxLogit := logits[0]

	for index, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
			maxIndex = index + 1
		}
	}

	return int32(maxIndex)
}

func topKSampleFloat32Native(
	logits []float32,
	temperature float32,
	topK int,
	seed uint64,
) int32 {
	elementCount := len(logits)

	if elementCount == 0 {
		return 0
	}

	k := topK

	if k <= 0 || k > elementCount {
		k = elementCount
	}

	probabilities, indices := softmaxAndSortNative(logits, temperature)
	cumulative := float32(0)

	for index := 0; index < k; index++ {
		cumulative += probabilities[index]
	}

	for index := 0; index < k; index++ {
		probabilities[index] /= cumulative
	}

	for index := k; index < len(probabilities); index++ {
		probabilities[index] = 0
	}

	rng := newSamplingRNG(seed)

	return int32(indices[drawFrom(probabilities, rng)])
}

func topPSampleFloat32Native(
	logits []float32,
	temperature float32,
	topP float32,
	seed uint64,
) int32 {
	elementCount := len(logits)

	if elementCount == 0 {
		return 0
	}

	probabilities, indices := softmaxAndSortNative(logits, temperature)
	cumulative := float32(0)
	cutoff := len(probabilities)

	for index := 0; index < len(probabilities); index++ {
		cumulative += probabilities[index]

		if cumulative >= topP {
			cutoff = index + 1
			break
		}
	}

	cumulative = 0

	for index := 0; index < cutoff; index++ {
		cumulative += probabilities[index]
	}

	for index := 0; index < cutoff; index++ {
		probabilities[index] /= cumulative
	}

	for index := cutoff; index < len(probabilities); index++ {
		probabilities[index] = 0
	}

	rng := newSamplingRNG(seed)

	return int32(indices[drawFrom(probabilities, rng)])
}

func samplingSoftmaxRowNative(logits, probabilities []float32, temperature float32) {
	if temperature == 0 {
		temperature = 1
	}

	for index, value := range logits {
		probabilities[index] = value / temperature
	}

	maximum := probabilities[0]

	for _, value := range probabilities[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var denominator float64

	for index, value := range probabilities {
		shifted := math.Exp(float64(value - maximum))
		probabilities[index] = float32(shifted)
		denominator += shifted
	}

	if denominator == 0 {
		return
	}

	scale := float32(1.0 / denominator)

	for index := range probabilities {
		probabilities[index] *= scale
	}
}
