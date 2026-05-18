//go:build arm64

package kernels

import "math"

func dropoutFloat32Native(
	dst, src []float32,
	seedState *[4]uint32,
	keepProb float32,
) {
	elementCount := len(src)
	blockCount := elementCount &^ 3
	scale := float32(1.0 / keepProb)
	threshold := math.Float32frombits(uint32(float64(keepProb) * (1 << 32)))

	if blockCount > 0 {
		dropoutFloat32NEONAsm(&dst[0], &src[0], blockCount, &seedState[0], scale, threshold)
	}

	for index := blockCount; index < elementCount; index++ {
		dst[index] = dropoutFloat32ScalarLane(
			src[index], seedState, scale, threshold,
		)
	}
}

func dropoutFloat32ScalarLane(
	value float32,
	seedState *[4]uint32,
	scale, threshold float32,
) float32 {
	randValue := dropoutXorshift32(&seedState[0])
	thresholdBits := math.Float32bits(threshold)

	if randValue >= thresholdBits {
		return 0
	}

	return value * scale
}

func dropoutXorshift32(seedLane *uint32) uint32 {
	value := *seedLane
	value ^= value << 13
	value ^= value >> 17
	value ^= value << 5
	*seedLane = value

	return value
}
