//go:build 386

package activation

import "golang.org/x/sys/cpu"

func LeakyReLUSlopeF32AVX2(dst, src *float32, count int, negativeSlope float32)
func LeakyReLUSlopeF32SSE2(dst, src *float32, count int, negativeSlope float32)
func PReLUF32AVX2(dst, src *float32, count int, negativeSlope float32)
func PReLUF32SSE2(dst, src *float32, count int, negativeSlope float32)
func ThresholdF32AVX2(dst, src *float32, count int, threshold float32)
func ThresholdF32SSE2(dst, src *float32, count int, threshold float32)
func HardTanhRangeF32AVX2(dst, src *float32, count int, minVal, maxVal float32)
func HardTanhRangeF32SSE2(dst, src *float32, count int, minVal, maxVal float32)
func ELUAlphaF32AVX2(dst, src *float32, count int, alpha float32)
func ELUAlphaF32SSE2(dst, src *float32, count int, alpha float32)
func CELUAlphaF32AVX2(dst, src *float32, count int, alpha float32)
func CELUAlphaF32SSE2(dst, src *float32, count int, alpha float32)
func HardShrinkF32AVX2(dst, src *float32, count int, lambda float32)
func HardShrinkF32SSE2(dst, src *float32, count int, lambda float32)
func SoftShrinkF32AVX2(dst, src *float32, count int, lambda float32)
func SoftShrinkF32SSE2(dst, src *float32, count int, lambda float32)
func SnakeF32AVX2(dst, src *float32, count int, alpha float32)
func SnakeF32SSE2(dst, src *float32, count int, alpha float32)
func SnakeParametricF32AVX2(dst, src *float32, count int, alpha, beta float32)
func SnakeParametricF32SSE2(dst, src *float32, count int, alpha, beta float32)
	func RReLUF32SSE2(dst, src *float32, count int, lower, upper float32)
func PReLUVF32AVX2(dst, src, slopes *float32, count int)
func PReLUVF32SSE2(dst, src, slopes *float32, count int)

var (
	leakyReLUSlopeF32Funcs = []paramSlopeKernelImpl{
		{LeakyReLUSlopeF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LeakyReLUSlopeF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LeakyReLUSlopeF32Generic, "generic", true},
	}
	preluF32Funcs = []paramSlopeKernelImpl{
		{PReLUF32AVX2, "avx2", cpu.X86.HasAVX2},
		{PReLUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{PReLUF32Generic, "generic", true},
	}
	thresholdF32Funcs = []paramSlopeKernelImpl{
		{ThresholdF32AVX2, "avx2", cpu.X86.HasAVX2},
		{ThresholdF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ThresholdF32Generic, "generic", true},
	}
	hardTanhRangeF32Funcs = []paramRangeKernelImpl{
		{HardTanhRangeF32AVX2, "avx2", cpu.X86.HasAVX2},
		{HardTanhRangeF32SSE2, "sse2", cpu.X86.HasSSE2},
		{HardTanhRangeF32Generic, "generic", true},
	}
	eluAlphaF32Funcs = []paramSlopeKernelImpl{
		{ELUAlphaF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{ELUAlphaF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ELUAlphaF32Generic, "generic", true},
	}
	celuAlphaF32Funcs = []paramSlopeKernelImpl{
		{CELUAlphaF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{CELUAlphaF32SSE2, "sse2", cpu.X86.HasSSE2},
		{CELUAlphaF32Generic, "generic", true},
	}
	hardShrinkF32Funcs = []paramSlopeKernelImpl{
		{HardShrinkF32AVX2, "avx2", cpu.X86.HasAVX2},
		{HardShrinkF32SSE2, "sse2", cpu.X86.HasSSE2},
		{HardShrinkF32Generic, "generic", true},
	}
	softShrinkF32Funcs = []paramSlopeKernelImpl{
		{SoftShrinkF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SoftShrinkF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SoftShrinkF32Generic, "generic", true},
	}
	snakeF32Funcs = []paramSlopeKernelImpl{
		{SnakeF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SnakeF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SnakeF32Generic, "generic", true},
	}
	snakeParametricF32Funcs = []paramRangeKernelImpl{
		{SnakeParametricF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SnakeParametricF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SnakeParametricF32Generic, "generic", true},
	}
	rreluF32Funcs = []paramRReluKernelImpl{
		{RReLUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{RReLUF32Generic, "generic", true},
	}
	preluVF32Funcs = []paramIndexedKernelImpl{
		{PReLUVF32AVX2, "avx2", cpu.X86.HasAVX2},
		{PReLUVF32SSE2, "sse2", cpu.X86.HasSSE2},
		{PReLUVF32Generic, "generic", true},
	}
)
