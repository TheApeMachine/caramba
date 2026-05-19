//go:build amd64

package dropout

import "golang.org/x/sys/cpu"

func DropoutFloat32Native(
	dst, src []float32,
	seedState *[4]uint32,
	keepProb float32,
) {
	if len(src) == 0 {
		return
	}

	dropoutF32Kernel(&dst[0], &src[0], len(src), seedState, keepProb)
}

var dropoutF32Funcs = []f32DropoutKernelImpl{
	{DropoutF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{DropoutF32Generic, "generic", true},
}
