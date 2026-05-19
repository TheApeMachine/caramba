package sampling

import "math"

func GreedySampleFloat32Native(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}

	bestIndex := 0
	bestValue := logits[0]

	for index, value := range logits[1:] {
		if value > bestValue {
			bestValue = value
			bestIndex = index + 1
		}
	}

	return int32(bestIndex)
}

func SamplingSoftmaxRowNative(logits, out []float32, temperature float32) {
	if temperature == 0 {
		temperature = 1
	}

	if len(logits) == 0 {
		return
	}

	maximum := logits[0]

	for _, value := range logits[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var denominator float64

	for index, value := range logits {
		shifted := math.Exp(float64((value - maximum) / temperature))
		out[index] = float32(shifted)
		denominator += shifted
	}

	if denominator == 0 {
		return
	}

	scale := float32(1.0 / denominator)

	for index := range out {
		out[index] *= scale
	}
}

func TopKSampleFloat32Native(logits []float32, temperature float32, topK int, seed uint64) int32 {
	sorted, indices := softmaxAndSort(logits, temperature)

	if topK <= 0 || topK > len(sorted) {
		topK = len(sorted)
	}

	var sum float32

	for index := 0; index < topK; index++ {
		sum += sorted[index]
	}

	if sum == 0 {
		return int32(indices[0])
	}

	rng := newSamplingRNG(seed)
	target := rng.Float32() * sum
	cumulative := float32(0)

	for index := 0; index < topK; index++ {
		cumulative += sorted[index]

		if cumulative >= target {
			return int32(indices[index])
		}
	}

	return int32(indices[topK-1])
}

func TopPSampleFloat32Native(logits []float32, temperature float32, topP float32, seed uint64) int32 {
	sorted, indices := softmaxAndSort(logits, temperature)

	if topP <= 0 {
		topP = 1
	}

	if topP > 1 {
		topP = 1
	}

	prefixLength := len(sorted)
	cumulative := float32(0)

	for index, probability := range sorted {
		cumulative += probability

		if cumulative >= topP {
			prefixLength = index + 1
			break
		}
	}

	if prefixLength == 0 {
		prefixLength = 1
	}

	var sum float32

	for index := 0; index < prefixLength; index++ {
		sum += sorted[index]
	}

	if sum == 0 {
		return int32(indices[0])
	}

	rng := newSamplingRNG(seed)
	target := rng.Float32() * sum
	cumulative = 0

	for index := 0; index < prefixLength; index++ {
		cumulative += sorted[index]

		if cumulative >= target {
			return int32(indices[index])
		}
	}

	return int32(indices[prefixLength-1])
}
