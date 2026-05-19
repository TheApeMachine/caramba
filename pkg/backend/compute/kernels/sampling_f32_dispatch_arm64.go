//go:build arm64

package kernels

import "sort"

func greedySampleFloat32Native(logits []float32) int32 {
	elementCount := len(logits)

	if elementCount == 0 {
		return 0
	}

	bestIndex := 0
	bestValue := logits[0]
	blockCount := elementCount &^ 3

	for index := 0; index < blockCount; index += 4 {
		blockMax := reduceMaxFloat32NEONAsm(&logits[index], 4)

		if blockMax <= bestValue {
			continue
		}

		blockOffset := 0

		for lane := 0; lane < 4; lane++ {
			if logits[index+lane] == blockMax {
				blockOffset = lane
				break
			}
		}

		bestValue = blockMax
		bestIndex = index + blockOffset
	}

	for index := blockCount; index < elementCount; index++ {
		if logits[index] > bestValue {
			bestValue = logits[index]
			bestIndex = index
		}
	}

	return int32(bestIndex)
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

	working := borrowFloat32Buffer(elementCount)
	probabilities := borrowFloat32Buffer(elementCount)
	indices := make([]int, elementCount)

	defer releaseFloat32Buffer(working)
	defer releaseFloat32Buffer(probabilities)

	copy(working, logits)
	samplingSoftmaxRowNative(working, probabilities, temperature)

	for index := range indices {
		indices[index] = index
	}

	sort.SliceStable(indices, func(left, right int) bool {
		return probabilities[indices[left]] > probabilities[indices[right]]
	})

	cumulative := float32(0)

	for index := 0; index < k; index++ {
		cumulative += probabilities[indices[index]]
	}

	if cumulative == 0 {
		return int32(indices[0])
	}

	scale := float32(1.0 / cumulative)

	for index := 0; index < k; index++ {
		probabilities[indices[index]] *= scale
	}

	for index := k; index < elementCount; index++ {
		probabilities[indices[index]] = 0
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

	working := borrowFloat32Buffer(elementCount)
	probabilities := borrowFloat32Buffer(elementCount)
	indices := make([]int, elementCount)

	defer releaseFloat32Buffer(working)
	defer releaseFloat32Buffer(probabilities)

	copy(working, logits)
	samplingSoftmaxRowNative(working, probabilities, temperature)

	for index := range indices {
		indices[index] = index
	}

	sort.SliceStable(indices, func(left, right int) bool {
		return probabilities[indices[left]] > probabilities[indices[right]]
	})

	cumulative := float32(0)
	cutoff := elementCount

	for index := 0; index < elementCount; index++ {
		cumulative += probabilities[indices[index]]

		if cumulative >= topP {
			cutoff = index + 1
			break
		}
	}

	cumulative = 0

	for index := 0; index < cutoff; index++ {
		cumulative += probabilities[indices[index]]
	}

	if cumulative == 0 {
		return int32(indices[0])
	}

	scale := float32(1.0 / cumulative)

	for index := 0; index < cutoff; index++ {
		probabilities[indices[index]] *= scale
	}

	for index := cutoff; index < elementCount; index++ {
		probabilities[indices[index]] = 0
	}

	rng := newSamplingRNG(seed)

	return int32(indices[drawFrom(probabilities, rng)])
}

func samplingSoftmaxRowNative(logits, probabilities []float32, temperature float32) {
	if len(logits) == 0 {
		return
	}

	if temperature == 0 {
		temperature = 1
	}

	for index, value := range logits {
		probabilities[index] = value / temperature
	}

	maximum := reduceMaxFloat32Native(probabilities)
	denominator := softmaxRowFillExpsNative(probabilities, probabilities, maximum)
	normalizeRow(probabilities, denominator)
}
