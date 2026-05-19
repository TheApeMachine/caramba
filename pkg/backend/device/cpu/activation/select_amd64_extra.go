//go:build amd64

package activation

import "golang.org/x/sys/cpu"

var (
	hardSigmoidF32Funcs = []f32KernelImpl{
		{HardSigmoidF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{HardSigmoidF32AVX2, "avx2", cpu.X86.HasAVX2},
		{HardSigmoidF32SSE2, "sse2", cpu.X86.HasSSE2},
		{HardSigmoidF32Generic, "generic", true},
	}
	hardSwishF32Funcs = []f32KernelImpl{
		{HardSwishF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{HardSwishF32AVX2, "avx2", cpu.X86.HasAVX2},
		{HardSwishF32SSE2, "sse2", cpu.X86.HasSSE2},
		{HardSwishF32Generic, "generic", true},
	}
	hardTanhF32Funcs = []f32KernelImpl{
		{HardTanhF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{HardTanhF32AVX2, "avx2", cpu.X86.HasAVX2},
		{HardTanhF32SSE2, "sse2", cpu.X86.HasSSE2},
		{HardTanhF32Generic, "generic", true},
	}
	hardGeluF32Funcs = []f32KernelImpl{
		{HardGeluF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{HardGeluF32AVX2, "avx2", cpu.X86.HasAVX2},
		{HardGeluF32SSE2, "sse2", cpu.X86.HasSSE2},
		{HardGeluF32Generic, "generic", true},
	}
	quickGeluF32Funcs = []f32KernelImpl{
		{QuickGeluF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{QuickGeluF32AVX2, "avx2", cpu.X86.HasAVX2},
		{QuickGeluF32SSE2, "sse2", cpu.X86.HasSSE2},
		{QuickGeluF32Generic, "generic", true},
	}
	tanhShrinkF32Funcs = []f32KernelImpl{
		{TanhShrinkF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{TanhShrinkF32AVX2, "avx2", cpu.X86.HasAVX2},
		{TanhShrinkF32SSE2, "sse2", cpu.X86.HasSSE2},
		{TanhShrinkF32Generic, "generic", true},
	}
)
