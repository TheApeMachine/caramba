package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
)

func LeakyReLUSlopeF32Generic(dst, src *float32, count int, negativeSlope float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastLeakyReLUWithSlope32(source[index], negativeSlope)
	}
}

func PReLUVF32Generic(dst, src, slopes *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)
	slopeLane := unsafe.Slice(slopes, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastPReLU32(source[index], slopeLane[index])
	}
}

func PReLUF32Generic(dst, src *float32, count int, negativeSlope float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastPReLU32(source[index], negativeSlope)
	}
}

func ThresholdF32Generic(dst, src *float32, count int, threshold float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastThreshold32(source[index], threshold)
	}
}

func HardTanhRangeF32Generic(dst, src *float32, count int, minVal, maxVal float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastHardTanhRange32(source[index], minVal, maxVal)
	}
}

func ELUAlphaF32Generic(dst, src *float32, count int, alpha float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastELUWithAlpha32(source[index], alpha)
	}
}

func CELUAlphaF32Generic(dst, src *float32, count int, alpha float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastCELUWithAlpha32(source[index], alpha)
	}
}

func HardShrinkF32Generic(dst, src *float32, count int, lambda float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastHardShrink32(source[index], lambda)
	}
}

func SoftShrinkF32Generic(dst, src *float32, count int, lambda float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSoftShrink32(source[index], lambda)
	}
}

func SnakeF32Generic(dst, src *float32, count int, alpha float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSnake32(source[index], alpha)
	}
}

func SnakeParametricF32Generic(dst, src *float32, count int, alpha, beta float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSnakeParametric32(source[index], alpha, beta)
	}
}

func RReLUF32Generic(dst, src *float32, count int, lower, upper float32) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)
	randomState := uint32(0xA5A5A5A5) ^
		*(*uint32)(unsafe.Pointer(&lower)) ^
		*(*uint32)(unsafe.Pointer(&upper))

	for index := 0; index < count; index++ {
		value := source[index]

		if value > 0 {
			destination[index] = value
			continue
		}

		randomState = randomState*1664525 + 1013904223
		slope := lower + float32(randomState>>8)/float32(0xFFFFFF)*(upper-lower)
		destination[index] = math.FastRReLU32(value, slope)
	}
}
