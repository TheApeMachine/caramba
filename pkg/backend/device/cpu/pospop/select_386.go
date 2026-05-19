package pospop

import "golang.org/x/sys/cpu"

func Count8AVX2(counts *[8]int, buf []byte)
func Count8SSE2(counts *[8]int, buf []byte)

func Count16AVX2(counts *[16]int, buf []uint16)
func Count16SSE2(counts *[16]int, buf []uint16)

func Count32AVX2(counts *[32]int, buf []uint32)
func Count32SSE2(counts *[32]int, buf []uint32)

func Count64SSE2(counts *[64]int, buf []uint64)
func Count64AVX2(counts *[64]int, buf []uint64)

var count8funcs = []count8impl{
	{Count8AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasBMI2},
	{Count8SSE2, "sse2", cpu.X86.HasSSE2},
	{Count8Generic, "generic", true},
}

var count16funcs = []count16impl{
	{Count16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasBMI2},
	{Count16SSE2, "sse2", cpu.X86.HasSSE2},
	{Count16Generic, "generic", true},
}

var count32funcs = []count32impl{
	{Count32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasBMI2},
	{Count32SSE2, "sse2", cpu.X86.HasSSE2},
	{Count32Generic, "generic", true},
}

var count64funcs = []count64impl{
	{Count64AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasBMI2},
	{Count64SSE2, "sse2", cpu.X86.HasSSE2},
	{Count64Generic, "generic", true},
}
