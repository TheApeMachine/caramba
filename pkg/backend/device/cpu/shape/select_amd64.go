//go:build amd64

package shape

import "golang.org/x/sys/cpu"

var copyContiguousF32Funcs = []f32CopyContiguousKernelImpl{
	{CopyContiguousF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{copyContiguousF32Generic, "generic", true},
}

var whereF32Funcs = []f32WhereKernelImpl{
	{whereF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{whereF32Generic, "generic", true},
}

var maskedFillF32Funcs = []f32MaskedFillKernelImpl{
	{maskedFillF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{maskedFillF32Generic, "generic", true},
}

func whereF32AVX512(dst, positive, negative *float32, mask []byte, count int) {
	WhereF32AVX512(dst, positive, negative, mask, count)
}

func maskedFillF32AVX512(dst, input *float32, fill float32, mask []byte, count int) {
	MaskedFillF32AVX512(dst, input, fill, mask, count)
}
