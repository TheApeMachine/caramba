package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/manifesto/dtype"
)

func PReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	if format == dtype.Float32 {
		preluF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			negativeSlope,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastPReLU32(value, negativeSlope)
	})
}

func PReLUV(
	dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int,
) {
	if format == dtype.Float32 {
		if slopeCount == 1 {
			preluF32Kernel(
				(*float32)(dst),
				(*float32)(src),
				count,
				loadF32(slopes, 0),
			)
			return
		}

		if slopeCount == count {
			preluVF32Kernel(
				(*float32)(dst),
				(*float32)(src),
				(*float32)(slopes),
				count,
			)
			return
		}
	}

	dispatchActivationLaneIndexed(dst, src, slopes, count, format, slopeCount,
		func(value, slope float32) float32 {
			return math.FastPReLU32(value, slope)
		},
	)
}

func LeakyReLUSlope(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	if format == dtype.Float32 {
		leakyReLUSlopeF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			negativeSlope,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastLeakyReLUWithSlope32(value, negativeSlope)
	})
}

func ELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	if format == dtype.Float32 {
		eluAlphaF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			alpha,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastELUWithAlpha32(value, alpha)
	})
}

func CELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	if format == dtype.Float32 {
		celuAlphaF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			alpha,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastCELUWithAlpha32(value, alpha)
	})
}

func Threshold(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	threshold float32,
) {
	if format == dtype.Float32 {
		thresholdF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			threshold,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastThreshold32(value, threshold)
	})
}

func HardTanhRange(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	minVal, maxVal float32,
) {
	if format == dtype.Float32 {
		hardTanhRangeF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			minVal,
			maxVal,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastHardTanhRange32(value, minVal, maxVal)
	})
}

func Snake(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	if format == dtype.Float32 {
		snakeF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			alpha,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastSnake32(value, alpha)
	})
}

func HardShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	if format == dtype.Float32 {
		hardShrinkF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			lambda,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastHardShrink32(value, lambda)
	})
}

func SoftShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	if format == dtype.Float32 {
		softShrinkF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			lambda,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastSoftShrink32(value, lambda)
	})
}

func RReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lower, upper float32,
) {
	if format == dtype.Float32 {
		rreluF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			lower,
			upper,
		)
		return
	}

	randomState := uint32(0xA5A5A5A5) ^
		*(*uint32)(unsafe.Pointer(&lower)) ^
		*(*uint32)(unsafe.Pointer(&upper))

	state := randomState

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		if value > 0 {
			return value
		}

		state = state*1664525 + 1013904223
		slope := lower + float32(state>>8)/float32(0xFFFFFF)*(upper-lower)

		return math.FastRReLU32(value, slope)
	})
}

func SnakeParametric(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha, beta float32,
) {
	if format == dtype.Float32 {
		snakeParametricF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
			alpha,
			beta,
		)
		return
	}

	dispatchActivationLane(dst, src, count, format, func(value float32) float32 {
		return math.FastSnakeParametric32(value, alpha, beta)
	})
}
