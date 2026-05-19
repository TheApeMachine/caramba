//go:build amd64

package activation

import "golang.org/x/sys/cpu"

func ExpF32AVX512(dst, src *float32, count int)
func ExpF32AVX2(dst, src *float32, count int)
func ExpF32SSE2(dst, src *float32, count int)
func LogF32AVX512(dst, src *float32, count int)
func LogF32AVX2(dst, src *float32, count int)
func LogF32SSE2(dst, src *float32, count int)
func Log1pF32AVX512(dst, src *float32, count int)
func Log1pF32AVX2(dst, src *float32, count int)
func Log1pF32SSE2(dst, src *float32, count int)
func Expm1F32AVX512(dst, src *float32, count int)
func Expm1F32AVX2(dst, src *float32, count int)
func Expm1F32SSE2(dst, src *float32, count int)
func SigmoidF32AVX512(dst, src *float32, count int)
func SigmoidF32AVX2(dst, src *float32, count int)
func SigmoidF32SSE2(dst, src *float32, count int)
func LogSigmoidF32AVX512(dst, src *float32, count int)
func LogSigmoidF32AVX2(dst, src *float32, count int)
func LogSigmoidF32SSE2(dst, src *float32, count int)
func TanhF32AVX512(dst, src *float32, count int)
func TanhF32AVX2(dst, src *float32, count int)
func TanhF32SSE2(dst, src *float32, count int)
func SiluF32AVX512(dst, src *float32, count int)
func SiluF32AVX2(dst, src *float32, count int)
func SiluF32SSE2(dst, src *float32, count int)
func GeluTanhF32AVX512(dst, src *float32, count int)
func GeluTanhF32AVX2(dst, src *float32, count int)
func GeluTanhF32SSE2(dst, src *float32, count int)
func GeluF32AVX512(dst, src *float32, count int)
func GeluF32AVX2(dst, src *float32, count int)
func GeluF32SSE2(dst, src *float32, count int)
func ReLUF32AVX512(dst, src *float32, count int)
func ReLUF32AVX2(dst, src *float32, count int)
func ReLUF32SSE2(dst, src *float32, count int)
func LeakyReLUF32AVX512(dst, src *float32, count int)
func LeakyReLUF32AVX2(dst, src *float32, count int)
func LeakyReLUF32SSE2(dst, src *float32, count int)
func ELUF32AVX512(dst, src *float32, count int)
func ELUF32AVX2(dst, src *float32, count int)
func ELUF32SSE2(dst, src *float32, count int)
func CELUF32AVX512(dst, src *float32, count int)
func CELUF32AVX2(dst, src *float32, count int)
func CELUF32SSE2(dst, src *float32, count int)
func SELUF32AVX512(dst, src *float32, count int)
func SELUF32AVX2(dst, src *float32, count int)
func SELUF32SSE2(dst, src *float32, count int)
func SoftplusF32AVX512(dst, src *float32, count int)
func SoftplusF32AVX2(dst, src *float32, count int)
func SoftplusF32SSE2(dst, src *float32, count int)
func MishF32AVX512(dst, src *float32, count int)
func MishF32AVX2(dst, src *float32, count int)
func MishF32SSE2(dst, src *float32, count int)
func SoftsignF32AVX512(dst, src *float32, count int)
func SoftsignF32AVX2(dst, src *float32, count int)
func SoftsignF32SSE2(dst, src *float32, count int)
func HardSigmoidF32AVX512(dst, src *float32, count int)
func HardSigmoidF32AVX2(dst, src *float32, count int)
func HardSigmoidF32SSE2(dst, src *float32, count int)
func HardSwishF32AVX512(dst, src *float32, count int)
func HardSwishF32AVX2(dst, src *float32, count int)
func HardSwishF32SSE2(dst, src *float32, count int)
func HardTanhF32AVX512(dst, src *float32, count int)
func HardTanhF32AVX2(dst, src *float32, count int)
func HardTanhF32SSE2(dst, src *float32, count int)
func HardGeluF32AVX512(dst, src *float32, count int)
func HardGeluF32AVX2(dst, src *float32, count int)
func HardGeluF32SSE2(dst, src *float32, count int)
func QuickGeluF32AVX512(dst, src *float32, count int)
func QuickGeluF32AVX2(dst, src *float32, count int)
func QuickGeluF32SSE2(dst, src *float32, count int)
func TanhShrinkF32AVX512(dst, src *float32, count int)
func TanhShrinkF32AVX2(dst, src *float32, count int)
func TanhShrinkF32SSE2(dst, src *float32, count int)

var (
	expF32Funcs = []f32KernelImpl{
		{ExpF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{ExpF32AVX2, "avx2", cpu.X86.HasAVX2},
		{ExpF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ExpF32Generic, "generic", true},
	}
	logF32Funcs = []f32KernelImpl{
		{LogF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{LogF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LogF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LogF32Generic, "generic", true},
	}
	log1pF32Funcs = []f32KernelImpl{
		{Log1pF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{Log1pF32AVX2, "avx2", cpu.X86.HasAVX2},
		{Log1pF32SSE2, "sse2", cpu.X86.HasSSE2},
		{Log1pF32Generic, "generic", true},
	}
	expm1F32Funcs = []f32KernelImpl{
		{Expm1F32AVX512, "avx512", cpu.X86.HasAVX512F},
		{Expm1F32AVX2, "avx2", cpu.X86.HasAVX2},
		{Expm1F32SSE2, "sse2", cpu.X86.HasSSE2},
		{Expm1F32Generic, "generic", true},
	}
	sigmoidF32Funcs = []f32KernelImpl{
		{SigmoidF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SigmoidF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SigmoidF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SigmoidF32Generic, "generic", true},
	}
	logSigmoidF32Funcs = []f32KernelImpl{
		{LogSigmoidF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{LogSigmoidF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LogSigmoidF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LogSigmoidF32Generic, "generic", true},
	}
	tanhF32Funcs = []f32KernelImpl{
		{TanhF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{TanhF32AVX2, "avx2", cpu.X86.HasAVX2},
		{TanhF32SSE2, "sse2", cpu.X86.HasSSE2},
		{TanhF32Generic, "generic", true},
	}
	siluF32Funcs = []f32KernelImpl{
		{SiluF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SiluF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SiluF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SiluF32Generic, "generic", true},
	}
	geluTanhF32Funcs = []f32KernelImpl{
		{GeluTanhF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{GeluTanhF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GeluTanhF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GeluTanhF32Generic, "generic", true},
	}
	geluF32Funcs = []f32KernelImpl{
		{GeluF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{GeluF32AVX2, "avx2", cpu.X86.HasAVX2},
		{GeluF32SSE2, "sse2", cpu.X86.HasSSE2},
		{GeluF32Generic, "generic", true},
	}
	reluF32Funcs = []f32KernelImpl{
		{ReLUF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{ReLUF32AVX2, "avx2", cpu.X86.HasAVX2},
		{ReLUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ReLUF32Generic, "generic", true},
	}
	leakyReluF32Funcs = []f32KernelImpl{
		{LeakyReLUF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{LeakyReLUF32AVX2, "avx2", cpu.X86.HasAVX2},
		{LeakyReLUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{LeakyReLUF32Generic, "generic", true},
	}
	eluF32Funcs = []f32KernelImpl{
		{ELUF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{ELUF32AVX2, "avx2", cpu.X86.HasAVX2},
		{ELUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ELUF32Generic, "generic", true},
	}
	celuF32Funcs = []f32KernelImpl{
		{CELUF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{CELUF32AVX2, "avx2", cpu.X86.HasAVX2},
		{CELUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{CELUF32Generic, "generic", true},
	}
	seluF32Funcs = []f32KernelImpl{
		{SELUF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SELUF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SELUF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SELUF32Generic, "generic", true},
	}
	softplusF32Funcs = []f32KernelImpl{
		{SoftplusF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SoftplusF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SoftplusF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SoftplusF32Generic, "generic", true},
	}
	mishF32Funcs = []f32KernelImpl{
		{MishF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MishF32AVX2, "avx2", cpu.X86.HasAVX2},
		{MishF32SSE2, "sse2", cpu.X86.HasSSE2},
		{MishF32Generic, "generic", true},
	}
	softsignF32Funcs = []f32KernelImpl{
		{SoftsignF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SoftsignF32AVX2, "avx2", cpu.X86.HasAVX2},
		{SoftsignF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SoftsignF32Generic, "generic", true},
	}
)
