//go:build amd64

package checkpoint

import "golang.org/x/sys/cpu"

var encodeFloat32DataFuncs = []float32DataEncodeKernelImpl{
	{checkpointEncodeFloat32DataAVX512, "avx512", cpu.X86.HasAVX512F},
	{encodeFloat32DataScalar, "generic", true},
}

var decodeFloat32DataFuncs = []float32DataDecodeKernelImpl{
	{checkpointDecodeFloat32DataAVX512, "avx512", cpu.X86.HasAVX512F},
	{decodeFloat32DataScalar, "generic", true},
}

func checkpointEncodeFloat32DataAVX512(dst []byte, src []float32) {
	CheckpointEncodeFloat32DataAVX512(&dst[0], &src[0], len(src))
}

func checkpointDecodeFloat32DataAVX512(dst []float32, src []byte) {
	CheckpointDecodeFloat32DataAVX512(&dst[0], &src[0], len(dst))
}
