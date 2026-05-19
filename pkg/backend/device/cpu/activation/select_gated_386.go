//go:build 386

package activation

import "golang.org/x/sys/cpu"

func SwiGLUTensorsF32AVX2(dst, gate, up *float32, count int)
func SwiGLUTensorsF32SSE2(dst, gate, up *float32, count int)
func LinGLUTensorsF32AVX2(dst, gate, up *float32, count int)
func LinGLUTensorsF32SSE2(dst, gate, up *float32, count int)
func ReGLUTensorsF32AVX2(dst, gate, up *float32, count int)
func ReGLUTensorsF32SSE2(dst, gate, up *float32, count int)
func GLUTensorsF32AVX2(dst, gate, up *float32, count int)
func GLUTensorsF32SSE2(dst, gate, up *float32, count int)
func SiGLUTensorsF32AVX2(dst, gate, up *float32, count int)
func SiGLUTensorsF32SSE2(dst, gate, up *float32, count int)
func SeGLUTensorsF32AVX2(dst, gate, up *float32, count int)
func SeGLUTensorsF32SSE2(dst, gate, up *float32, count int)
func GeGLUTensorsF32AVX2(dst, gate, up *float32, count int)
func GeGLUTensorsF32SSE2(dst, gate, up *float32, count int)
func GeGLUTanhTensorsF32AVX2(dst, gate, up *float32, count int)
func GeGLUTanhTensorsF32SSE2(dst, gate, up *float32, count int)

var (
	swiGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{SwiGLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SwiGLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SwiGLUTensorsF32Generic, "generic", true},
	}
	linGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{LinGLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LinGLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LinGLUTensorsF32Generic, "generic", true},
	}
	reGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{ReGLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{ReGLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ReGLUTensorsF32Generic, "generic", true},
	}
	gluTensorsFuncs = []gatedTensorsKernelImpl{
		{GLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GLUTensorsF32Generic, "generic", true},
	}
	siGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{SiGLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SiGLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SiGLUTensorsF32Generic, "generic", true},
	}
	seGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{SeGLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SeGLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SeGLUTensorsF32Generic, "generic", true},
	}
	geGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{GeGLUTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GeGLUTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GeGLUTensorsF32Generic, "generic", true},
	}
	geGLUTanhTensorsFuncs = []gatedTensorsKernelImpl{
		{GeGLUTanhTensorsF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GeGLUTanhTensorsF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GeGLUTanhTensorsF32Generic, "generic", true},
	}
)
