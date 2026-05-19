package dropout

import (
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func Dropout(
	dst, src unsafe.Pointer,
	count int,
	config DropoutConfig,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	if format != dtype.Float32 {
		panic("dropout: only dtype.Float32 is implemented")
	}

	runDropoutF32(dst, src, count, config)
}

func DropoutSeedState(seed uint64) [4]uint32 {
	return [4]uint32{
		uint32(seed),
		uint32(seed >> 32),
		uint32(seed ^ 0x9e3779b9),
		uint32((seed >> 32) ^ 0x6c078965),
	}
}

func dropoutXorshift32(seedLane *uint32) uint32 {
	value := *seedLane
	value ^= value << 13
	value ^= value >> 17
	value ^= value << 5
	*seedLane = value

	return value
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

func dropoutThreshold(keepProb float32) float32 {
	return math.Float32frombits(uint32(float64(keepProb) * (1 << 32)))
}
