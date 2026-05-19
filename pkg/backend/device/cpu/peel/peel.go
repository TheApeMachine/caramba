package peel

import "unsafe"

/*
SimdLaneCount is the f32 element alignment required before calling an
amd64 SIMD unary kernel; tails run through the scalar reference in Go.
*/
func SimdLaneCount(isaName string) int {
	switch isaName {
	case "avx512":
		return 16
	case "avx2":
		return 8
	case "sse2":
		return 4
	default:
		return 1
	}
}

/*
WrapF32Unary runs the SIMD kernel on a lane-aligned prefix and the generic
reference on any tail lanes.
*/
func WrapF32Unary(
	simdKernel func(dst, src *float32, count int),
	genericKernel func(dst, src *float32, count int),
	isaName string,
) func(dst, src *float32, count int) {
	align := SimdLaneCount(isaName)

	return func(dst, src *float32, count int) {
		if count <= 0 {
			return
		}

		headCount := count - count%align
		if headCount > 0 {
			simdKernel(dst, src, headCount)
		}

		tailCount := count - headCount
		if tailCount > 0 {
			genericKernel(
				offsetFloat32Pointer(dst, headCount),
				offsetFloat32Pointer(src, headCount),
				tailCount,
			)
		}
	}
}

/*
WrapGatedTensors runs gated tensor SIMD on an aligned prefix and generic on the tail.
*/
func WrapGatedTensors(
	simdKernel func(dst, gate, up *float32, count int),
	genericKernel func(dst, gate, up *float32, count int),
	isaName string,
) func(dst, gate, up *float32, count int) {
	align := SimdLaneCount(isaName)

	return func(dst, gate, up *float32, count int) {
		if count <= 0 {
			return
		}

		headCount := count - count%align
		if headCount > 0 {
			simdKernel(dst, gate, up, headCount)
		}

		tailCount := count - headCount
		if tailCount > 0 {
			genericKernel(
				offsetFloat32Pointer(dst, headCount),
				offsetFloat32Pointer(gate, headCount),
				offsetFloat32Pointer(up, headCount),
				tailCount,
			)
		}
	}
}

/*
WrapParamSlope peels count for parametric slope kernels.
*/
func WrapParamSlope(
	simdKernel func(dst, src *float32, count int, slope float32),
	genericKernel func(dst, src *float32, count int, slope float32),
	isaName string,
) func(dst, src *float32, count int, slope float32) {
	align := SimdLaneCount(isaName)

	return func(dst, src *float32, count int, slope float32) {
		if count <= 0 {
			return
		}

		headCount := count - count%align
		if headCount > 0 {
			simdKernel(dst, src, headCount, slope)
		}

		tailCount := count - headCount
		if tailCount > 0 {
			genericKernel(
				offsetFloat32Pointer(dst, headCount),
				offsetFloat32Pointer(src, headCount),
				tailCount,
				slope,
			)
		}
	}
}

/*
WrapParamRange peels count for two-scalar param kernels.
*/
func WrapParamRange(
	simdKernel func(dst, src *float32, count int, minVal, maxVal float32),
	genericKernel func(dst, src *float32, count int, minVal, maxVal float32),
	isaName string,
) func(dst, src *float32, count int, minVal, maxVal float32) {
	align := SimdLaneCount(isaName)

	return func(dst, src *float32, count int, minVal, maxVal float32) {
		if count <= 0 {
			return
		}

		headCount := count - count%align
		if headCount > 0 {
			simdKernel(dst, src, headCount, minVal, maxVal)
		}

		tailCount := count - headCount
		if tailCount > 0 {
			genericKernel(
				offsetFloat32Pointer(dst, headCount),
				offsetFloat32Pointer(src, headCount),
				tailCount,
				minVal,
				maxVal,
			)
		}
	}
}

/*
WrapParamRRelu peels count for RReLU range kernels.
*/
func WrapParamRRelu(
	simdKernel func(dst, src *float32, count int, lower, upper float32),
	genericKernel func(dst, src *float32, count int, lower, upper float32),
	isaName string,
) func(dst, src *float32, count int, lower, upper float32) {
	align := SimdLaneCount(isaName)

	return func(dst, src *float32, count int, lower, upper float32) {
		if count <= 0 {
			return
		}

		headCount := count - count%align
		if headCount > 0 {
			simdKernel(dst, src, headCount, lower, upper)
		}

		tailCount := count - headCount
		if tailCount > 0 {
			genericKernel(
				offsetFloat32Pointer(dst, headCount),
				offsetFloat32Pointer(src, headCount),
				tailCount,
				lower,
				upper,
			)
		}
	}
}

/*
WrapParamIndexed peels count for PReLU vector-slope kernels.
*/
func WrapParamIndexed(
	simdKernel func(dst, src, slopes *float32, count int),
	genericKernel func(dst, src, slopes *float32, count int),
	isaName string,
) func(dst, src, slopes *float32, count int) {
	align := SimdLaneCount(isaName)

	return func(dst, src, slopes *float32, count int) {
		if count <= 0 {
			return
		}

		headCount := count - count%align
		if headCount > 0 {
			simdKernel(dst, src, slopes, headCount)
		}

		tailCount := count - headCount
		if tailCount > 0 {
			genericKernel(
				offsetFloat32Pointer(dst, headCount),
				offsetFloat32Pointer(src, headCount),
				offsetFloat32Pointer(slopes, headCount),
				tailCount,
			)
		}
	}
}

func offsetFloat32Pointer(base *float32, elementOffset int) *float32 {
	if elementOffset == 0 {
		return base
	}

	byteOffset := elementOffset * int(unsafe.Sizeof(float32(0)))

	return (*float32)(unsafe.Add(unsafe.Pointer(base), byteOffset))
}
