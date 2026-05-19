//go:build 386

package activation

import "golang.org/x/sys/cpu"

func SwiGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func SwiGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func LinGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func LinGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func ReGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func ReGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func GLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func GLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func SiGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func SiGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func SeGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func SeGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func GeGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func GeGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
func GeGLUTanhPackedF32AVX2(dst, packed *float32, batch, halfCount int)
func GeGLUTanhPackedF32SSE2(dst, packed *float32, batch, halfCount int)

var (
	swiGLUPackedFuncs = []gatedPackedKernelImpl{
		{SwiGLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SwiGLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SwiGLUPackedF32Generic, "generic", true},
	}
	linGLUPackedFuncs = []gatedPackedKernelImpl{
		{LinGLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LinGLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LinGLUPackedF32Generic, "generic", true},
	}
	reGLUPackedFuncs = []gatedPackedKernelImpl{
		{ReGLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{ReGLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ReGLUPackedF32Generic, "generic", true},
	}
	gluPackedFuncs = []gatedPackedKernelImpl{
		{GLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GLUPackedF32Generic, "generic", true},
	}
	siGLUPackedFuncs = []gatedPackedKernelImpl{
		{SiGLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SiGLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SiGLUPackedF32Generic, "generic", true},
	}
	seGLUPackedFuncs = []gatedPackedKernelImpl{
		{SeGLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SeGLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SeGLUPackedF32Generic, "generic", true},
	}
	geGLUPackedFuncs = []gatedPackedKernelImpl{
		{GeGLUPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GeGLUPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GeGLUPackedF32Generic, "generic", true},
	}
	geGLUTanhPackedFuncs = []gatedPackedKernelImpl{
		{GeGLUTanhPackedF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GeGLUTanhPackedF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GeGLUTanhPackedF32Generic, "generic", true},
	}
)
