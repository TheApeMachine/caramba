//go:build amd64

package activation

import "golang.org/x/sys/cpu"

func SoftmaxF32AVX512(dst, src *float32, count int)
func SoftmaxF32AVX2(dst, src *float32, count int)
func SoftmaxF32SSE2(dst, src *float32, count int)
func LogSoftmaxF32AVX512(dst, src *float32, count int)
func LogSoftmaxF32AVX2(dst, src *float32, count int)
func LogSoftmaxF32SSE2(dst, src *float32, count int)

var (
	softmaxF32Funcs = []f32KernelImpl{
		{SoftmaxF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SoftmaxF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SoftmaxF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SoftmaxF32Generic, "generic", true},
	}
	logSoftmaxF32Funcs = []f32KernelImpl{
		{LogSoftmaxF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{LogSoftmaxF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LogSoftmaxF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LogSoftmaxF32Generic, "generic", true},
	}
)
