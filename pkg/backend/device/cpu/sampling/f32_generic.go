package sampling

import (
	"math"
	"unsafe"
)

func greedySampleF32Generic(logits *float32, count int) int32 {
	return GreedySampleGeneric(unsafe.Slice(logits, count))
}

func samplingSoftmaxRowF32Generic(logits, out *float32, temperature float32, count int) {
	SamplingSoftmaxRowGeneric(
		unsafe.Slice(logits, count),
		unsafe.Slice(out, count),
		temperature,
	)
}

func GreedySampleGeneric(logits []float32) int32 {
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

func SamplingSoftmaxRowGeneric(logits, out []float32, temperature float32) {
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
